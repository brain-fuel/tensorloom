%% @doc Structural/scalar generator composition, shared across native frameworks.
%% The framework supplies every generator and shrink tree; no portable seed or
%% custom shrinker replaces PropEr, StreamData or qcheck.
%% ref:DEC-native-property-frameworks ref:DEC-structural-size-budget
-module(lawspec_beam_generators).
-export([generator/6, generator/7, with_cache/1, with_native/3, check_drawn/4]).

%% A chosen strategy cannot widen the law's input domain. Apply this to each
%% generated value and shrink, before any later input can discard the tuple.
%% ref:REQ-harness-units ref:DEC-never-pass-vacuously
check_drawn(_, _, _, '$lawspec_empty_domain') ->
    '$lawspec_empty_domain';
check_drawn(Strategy, Input, Predicate, Value) ->
    case Predicate(Value) of
        true -> Value;
        false -> lawspec_beam_harness:abort({strategy_outside_refinement, Strategy, Input, Value})
    end.

%% Factories live exclusively in test code. The canonical schema still owns
%% every logical contract; the native schema only converts representations.
%% ref:DEC-native-bindings-typed-identity
with_native(Schema, Native, Factories) ->
    maps:foreach(
        fun(Name, Factory) ->
            case maps:is_key(Name, maps:get(arities, Schema)) andalso is_function(Factory, 1) of
                true -> ok;
                false -> erlang:error({lawspec, {invalid_native_generator, Name}})
            end
        end,
        Factories
    ),
    Schema#{native_generators => Factories, native_generator_schema => Native}.

generator(F, T, Schema, Symbols, Bounds, Witnesses) ->
    generator(F, T, Schema, Symbols, Bounds, Witnesses, none).
generator(F, T, Schema, Symbols, Bounds, Witnesses, Index) ->
    F:sized(fun(Size) ->
        case factory(T, Schema) of
            {ok, Factory} -> custom(F, T, Schema, Symbols, Size + 1, Bounds, Index, Factory);
            error -> automatic(F, T, Schema, Symbols, Size, Bounds, Witnesses, Index)
        end
    end).

automatic(F, T, Schema, Symbols, Size, Bounds, Witnesses, Index) ->
    Limit = lists:max([64 | [value_nodes(V) || V <- Witnesses]]),
    Minimum = minimum(T, Schema, Limit),
    case Minimum of
        none ->
            erlang:error({lawspec, {no_structural_generator, T}});
        _ ->
            Budget = max(Minimum, Size + 1),
            case Index of
                {Target, Equations} ->
                    lawspec_beam_index:generator(
                        F,
                        T,
                        Schema,
                        Target,
                        Equations,
                        Budget,
                        fun(Child) -> minimum(Child, Schema, Limit) end,
                        fun(Child, Available) ->
                            build(F, Child, Schema, Symbols, Available, [])
                        end,
                        fun(Child, Value) -> accepted(Value, Child, Schema) end,
                        fun(Child, K, Available) ->
                            case factory(Child, Schema) of
                                error ->
                                    none;
                                {ok, Factory} ->
                                    {custom,
                                        custom(
                                            F,
                                            Child,
                                            Schema,
                                            Symbols,
                                            Available,
                                            [],
                                            {K, Equations},
                                            Factory
                                        )}
                            end
                        end
                    );
                none ->
                    Native = build(F, T, Schema, Symbols, Budget, Bounds),
                    with_witnesses(F, Native, Witnesses)
            end
    end.

factory({Name, _}, S) -> maps:find(Name, maps:get(native_generators, S, #{})).

custom(F, {Name, Args} = T, S, Symbols, Budget, Bounds, Index, Factory) ->
    Native = maps:get(native_generator_schema, S),
    Label = <<"native generator ", Name/binary>>,
    Children = [
        F:sized(fun(_) ->
            Available = max(1, Budget - 1),
            case minimum(Child, S, Available) of
                none ->
                    F:constrain(F:exactly('$lawspec_uninhabited'), fun(_) -> false end);
                _ ->
                    F:map(
                        build(F, Child, S, Symbols, Available, []),
                        fun(V) -> lawspec_beam_schema:to_native(V, Child, Native) end
                    )
            end
        end)
     || Child <- Args
    ],
    Checked = lawspec_beam_runtime:contextual(Label, fun() ->
        F:map(Factory(Children), fun(Value) ->
            lawspec_beam_runtime:contextual(Label, fun() ->
                lawspec_beam_schema:validate(
                    lawspec_beam_schema:from_native(Value, T, Native), T, S
                )
            end)
        end)
    end),
    %% Refinements constrain valid values, after validation of each sample and
    %% shrink. They must never hide a broken native factory or codec.
    case {Bounds, Index} of
        {[], none} ->
            Checked;
        _ ->
            F:refine_input(Checked, fun(V) ->
                lists:all(
                    fun
                        ({length, Op, Bound}) ->
                            lawspec_beam_scalar:binary(
                                Op,
                                lawspec_beam_scalar:helper(
                                    <<"length">>, [V], [Name], <<"Integer">>
                                ),
                                Bound,
                                <<"Integer">>,
                                <<"Integer">>
                            );
                        ({Op, Bound}) ->
                            lawspec_beam_scalar:binary(Op, V, Bound, Name, Name)
                    end,
                    Bounds
                ) andalso
                    case Index of
                        none -> true;
                        {Target, _} when Target < 0 -> true;
                        {Target, _} -> lawspec_beam_schema:index(V, T, 0, S) =:= Target
                    end
            end)
    end.

value_nodes({ls_data, _, Fields}) -> 1 + lists:sum([value_nodes(V) || V <- Fields]);
value_nodes({ls_presence, _, {some, V}}) -> 1 + value_nodes(V);
value_nodes(Vs) when is_list(Vs) -> 1 + lists:sum([1 + value_nodes(V) || V <- Vs]);
value_nodes(_) -> 1.

build(F, T, S, Symbols, Budget, Bounds) ->
    case factory(T, S) of
        {ok, Factory} -> custom(F, T, S, Symbols, Budget, Bounds, none, Factory);
        error -> build_native(F, T, S, Symbols, Budget, Bounds)
    end.

build_native(F, {<<"List">>, [T]}, S, Symbols, Budget, Bounds) ->
    Lengths = lengths(Bounds),
    Limit =
        case Lengths of
            [] -> Budget - 1;
            _ -> max(64, Budget - 1)
        end,
    case minimum(T, S, Limit) of
        none ->
            F:refine_input(F:exactly([]), fun(_) -> length_allowed(0, Lengths) end);
        Cost ->
            F:bind(length_generator(F, Lengths, (Budget - 1) div Cost), fun
                (0) ->
                    F:exactly([]);
                (Count) ->
                    F:vector(Count, build(F, T, S, Symbols, max(Cost, (Budget - 1) div Count), []))
            end)
    end;
build_native(F, {Name, [T]}, S, Symbols, Budget, _) when
    Name =:= <<"Optional">>; Name =:= <<"Nullable">>
->
    Empty = F:exactly({ls_presence, Name, none}),
    case minimum(T, S, Budget - 1) of
        none ->
            Empty;
        _ ->
            F:oneof([
                Empty,
                F:bind(
                    build(F, T, S, Symbols, Budget - 1, []),
                    fun(V) -> {ls_presence, Name, {some, V}} end
                )
            ])
    end;
build_native(F, T, S, Symbols, Budget, Bounds) ->
    case lawspec_beam_schema:constructors(T, S) of
        none ->
            scalar(F, T, maps:get(bits, S), Symbols, Bounds);
        Constructors ->
            Choices = [
                constructor(F, C, Allocation, S, Symbols)
             || Raw <- Constructors,
                C <- lawspec_beam_schema:witness_instances(Raw),
                Allocation <- [
                    allocation(
                        maps:get(fields, C), S, Budget - 1 - length(maps:get(witness_values, C))
                    )
                ],
                Allocation =/= none
            ],
            Canonical = F:map(F:oneof(Choices), fun(V) -> canonical(T, V) end),
            F:constrain(Canonical, fun(V) -> accepted(V, T, S) end)
    end.

constructor(F, C, Allocation, S, Symbols) ->
    Fields = maps:get(fields, C),
    constructor_fields(F, C, lists:zip(Fields, Allocation), S, Symbols, []).
constructor_fields(F, C, [], _, _, Values) ->
    F:exactly({ls_data, maps:get(tag, C), Values ++ maps:get(witness_values, C)});
constructor_fields(F, C, [{{_, T}, Budget} | Rest], S, Symbols, Values) ->
    {Bounds, Witnesses} =
        case maps:find(length(Values) + 1, maps:get(generators, C, #{})) of
            error -> {[], []};
            {ok, Constraints} -> Constraints(S, maps:get(arguments, C, []), Values)
        end,
    Native = build(F, T, S, Symbols, Budget, Bounds),
    Generator =
        case factory(T, S) of
            {ok, _} -> Native;
            error -> with_witnesses(F, Native, Witnesses)
        end,
    F:bind(Generator, fun(V) -> constructor_fields(F, C, Rest, S, Symbols, Values ++ [V]) end).

with_witnesses(_, Native, []) ->
    Native;
with_witnesses(F, Native, Witnesses) ->
    F:frequency([{1, F:oneof([F:exactly(V) || V <- Witnesses])}, {9, Native}]).

%% Sorting and deduplication map the native tree, so every shrink remains a
%% canonical collection. A custom native factory is validated without repair.
%% ref:DEC-native-property-frameworks ref:DEC-shrink-within-domain
canonical({<<"lawspec.collections::type::EqSet">>, _}, {ls_data, Tag, [Items]}) ->
    {ls_data, Tag, [distinct_equal(Items, [])]};
canonical({<<"lawspec.collections::type::Set">>, _}, {ls_data, Tag, [Items]}) ->
    {ls_data, Tag, [lists:usort(fun(A, B) -> lawspec_beam_scalar:compare(A, B) =< 0 end, Items)]};
canonical({<<"lawspec.collections::type::KeyVal">>, _}, {ls_data, Tag, [Entries]}) ->
    {ls_data, Tag, [
        lists:usort(
            fun({ls_data, _, [A, _]}, {ls_data, _, [B, _]}) ->
                lawspec_beam_scalar:compare(A, B) =< 0
            end,
            Entries
        )
    ]};
canonical(_, Value) ->
    Value.

distinct_equal([], Previous) ->
    lists:reverse(Previous);
distinct_equal([Item | Rest], Previous) ->
    case lists:any(fun(Prior) -> lawspec_beam_scalar:equal(Item, Prior) end, Previous) of
        true -> distinct_equal(Rest, Previous);
        false -> distinct_equal(Rest, [Item | Previous])
    end.

accepted(V, T, S) ->
    try
        lawspec_beam_schema:validate(V, T, S),
        true
    catch
        error:{lawspec, {refinement_violation, _}} -> false
    end.

%% The budget counts structural nodes, including scalar leaves. Searching the
%% finite budget terminates for recursive types without assuming a base case.
%% ref:DEC-structural-size-budget
minimum(_, _, Limit) when Limit < 1 -> none;
minimum(T, S, Limit) ->
    Key = {T, S, Limit},
    Cache =
        case get({?MODULE, minimum}) of
            undefined -> #{};
            Existing -> Existing
        end,
    case maps:find(Key, Cache) of
        {ok, Cost} ->
            Cost;
        error ->
            Cost = minimum(T, S, 1, Limit),
            %% Recursive calls have filled entries too; preserve them.
            Current =
                case get({?MODULE, minimum}) of
                    undefined -> #{};
                    Populated -> Populated
                end,
            put({?MODULE, minimum}, Current#{Key => Cost}),
            Cost
    end.
minimum(_, _, Cost, Limit) when Cost > Limit -> none;
minimum(T, S, Cost, Limit) ->
    case inhabited(T, S, Cost) of
        true -> Cost;
        false -> minimum(T, S, Cost + 1, Limit)
    end.

inhabited(_, _, Budget) when Budget < 1 -> false;
inhabited({Name, _}, _, _) when
    Name =:= <<"List">>; Name =:= <<"Optional">>; Name =:= <<"Nullable">>
->
    true;
inhabited(T, S, Budget) ->
    case factory(T, S) of
        {ok, _} -> true;
        error -> inhabited_native(T, S, Budget)
    end.
inhabited_native(T, S, Budget) ->
    case lawspec_beam_schema:constructors(T, S) of
        none ->
            true;
        Cs ->
            lists:any(
                fun(C) ->
                    allocation(
                        maps:get(fields, C),
                        S,
                        Budget - 1 - length(maps:get(witness_values, C))
                    ) =/= none
                end,
                lists:append([lawspec_beam_schema:witness_instances(C) || C <- Cs])
            )
    end.

allocation(Fields, S, Budget) ->
    case costs(Fields, S, Budget) of
        none ->
            none;
        [] ->
            [];
        Minimums ->
            Extra = Budget - lists:sum(Minimums),
            Count = length(Minimums),
            [
                Cost + Extra div Count +
                    case I =< Extra rem Count of
                        true -> 1;
                        false -> 0
                    end
             || {I, Cost} <- lists:enumerate(Minimums)
            ]
    end.
costs([], _, Budget) when Budget >= 0 -> [];
costs(_, _, Budget) when Budget < 1 -> none;
costs([{_, T} | Fields], S, Budget) ->
    case minimum(T, S, Budget) of
        none ->
            none;
        Cost ->
            case costs(Fields, S, Budget - Cost) of
                none -> none;
                Rest -> [Cost | Rest]
            end
    end.

scalar(F, {<<"Bool">>, []}, _, _, _) ->
    F:oneof([F:exactly(false), F:exactly(true)]);
scalar(F, {<<"Unit">>, []}, _, _, _) ->
    F:exactly(ls_unit);
scalar(F, {<<"Null">>, []}, _, _, _) ->
    F:exactly(ls_null);
scalar(F, {<<"Undefined">>, []}, _, _, _) ->
    F:exactly(ls_undefined);
scalar(F, {<<"Rational">>, []}, _, _, _) ->
    F:bind(F:fixed_list([F:integer(none, none), F:integer(1, none)]), fun([N, D]) ->
        lawspec_beam_scalar:ratio(N, D)
    end);
scalar(F, {<<"Decimal">>, []}, _, _, _) ->
    F:bind(
        F:fixed_list([F:integer(none, none), F:sized(fun(Size) -> F:integer(-Size, Size) end)]),
        fun([C, E]) -> lawspec_beam_scalar:decimal(C, E) end
    );
scalar(F, {<<"Float32">>, []}, _, _, _) ->
    ieee(F, 32);
scalar(F, {<<"Float64">>, []}, _, _, _) ->
    ieee(F, 64);
scalar(F, {Name, []}, _, _, _) when Name =:= <<"Complex64">>; Name =:= <<"Complex128">> ->
    W =
        case Name of
            <<"Complex64">> -> 32;
            _ -> 64
        end,
    F:bind(F:fixed_list([ieee(F, W), ieee(F, W)]), fun([R, I]) -> {ls_complex, W, R, I} end);
scalar(F, {<<"Char">>, []}, _, _, _) ->
    unicode_scalar(F);
scalar(F, {<<"CodePoint">>, []}, _, _, _) ->
    F:integer(0, 16#10ffff);
scalar(F, {<<"CodeUnit16">>, []}, _, _, _) ->
    F:integer(0, 16#ffff);
scalar(F, {<<"Text">>, []}, _, _, Bounds) ->
    F:bind(sequence(F, unicode_scalar(F), Bounds), fun unicode:characters_to_binary/1);
scalar(F, {<<"Bytes">>, []}, _, _, Bounds) ->
    case lengths(Bounds) of
        [] -> F:binary();
        _ -> F:bind(sequence(F, F:integer(0, 255), Bounds), fun list_to_binary/1)
    end;
scalar(F, {Name, []}, _, _, Bounds) when Name =:= <<"CodePointText">>; Name =:= <<"Utf16Text">> ->
    Max =
        case Name of
            <<"CodePointText">> -> 16#10ffff;
            _ -> 16#ffff
        end,
    F:bind(sequence(F, F:integer(0, Max), Bounds), fun(Units) -> {ls_raw, Name, Units} end);
scalar(F, {<<"Symbol">>, []}, _, Symbols, _) ->
    F:bind(F:integer(0, none), fun(N) ->
        Key = integer_to_binary(N),
        {ls_symbol, {{Symbols, generated}, Key}, Key}
    end);
scalar(F, {Name, []}, Bits, _, Bounds) ->
    {Lo, Hi} = lists:foldl(fun limit/2, lawspec_beam_scalar:bounds(Name, Bits), Bounds),
    case Lo =/= none andalso Hi =/= none andalso Lo > Hi of
        true -> F:exactly('$lawspec_empty_domain');
        false -> F:integer(Lo, Hi)
    end.

ieee(F, Width) ->
    F:bind(
        F:integer(0, (1 bsl Width) - 1),
        fun(Bits) -> lawspec_beam_scalar:float_bits(Width, Bits) end
    ).
unicode_scalar(F) -> F:oneof([F:integer(0, 16#d7ff), F:integer(16#e000, 16#10ffff)]).

%% Native length draws and vectors preserve both element and length shrinking.
%% A required minimum raises the size floor, while an empty interval retries
%% the whole dependent tuple. Hints never replace a supplied native factory.
%% ref:DEC-shrink-within-domain ref:DEC-native-property-frameworks
lengths(Bounds) -> [{Op, V} || {length, Op, V} <- Bounds].
sequence(F, Element, Bounds) ->
    case lengths(Bounds) of
        [] ->
            F:list(Element);
        Lengths ->
            F:sized(fun(Size) ->
                F:bind(length_generator(F, Lengths, Size), fun(Count) ->
                    F:vector(Count, Element)
                end)
            end)
    end.
length_generator(F, Bounds, Size) ->
    {Lo, Hi} = lists:foldl(fun limit/2, {0, none}, Bounds),
    case Hi =/= none andalso Lo > Hi of
        true -> F:exactly('$lawspec_empty_domain');
        false -> F:integer(Lo, upper(Hi, max(Lo, Size)))
    end.
length_allowed(N, Bounds) ->
    {Lo, Hi} = lists:foldl(fun limit/2, {0, none}, Bounds),
    N >= Lo andalso (Hi =:= none orelse N =< Hi).

limit({Op, V}, {Lo, Hi}) ->
    {N, D} = lawspec_beam_scalar:exact(V),
    Floor =
        N div D -
            case N < 0 andalso N rem D =/= 0 of
                true -> 1;
                false -> 0
            end,
    Ceil =
        Floor +
            case N rem D =/= 0 of
                true -> 1;
                false -> 0
            end,
    case Op of
        <<">">> -> {lower(Lo, Floor + 1), Hi};
        <<">=">> -> {lower(Lo, Ceil), Hi};
        <<"<">> -> {Lo, upper(Hi, Ceil - 1)};
        <<"<=">> -> {Lo, upper(Hi, Floor)};
        <<"==">> -> {lower(Lo, Ceil), upper(Hi, Floor)};
        _ -> {Lo, Hi}
    end.
lower(none, N) -> N;
lower(L, N) -> max(L, N).
upper(none, N) -> N;
upper(H, N) -> min(H, N).

with_cache(Run) ->
    Previous = erase({?MODULE, minimum}),
    try
        lawspec_beam_index:with_cache(Run)
    after
        case Previous of
            undefined -> erase({?MODULE, minimum});
            _ -> put({?MODULE, minimum}, Previous)
        end
    end.
