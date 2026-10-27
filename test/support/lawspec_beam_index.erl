%% @doc Solve structural indices before generating native framework trees.
%% Reachability carries minimum node costs, so equal-index recursive branches
%% also terminate within the structural budget. Children may have a larger
%% index than their parent (subtraction, division, remainder and powers).
%% ref:DEC-indexed-families-as-evidence ref:DEC-structural-size-budget
-module(lawspec_beam_index).
-export([generator/9, generator/10, with_cache/1]).

generator(F, Type, Schema, Target, Equations, Budget, Minimum, Build, Accepted) ->
    generator(F, Type, Schema, Target, Equations, Budget, Minimum, Build, Accepted, fun(_, _, _) ->
        none
    end).
generator(F, Type, Schema, Target, Equations, Budget, Minimum, Build, Accepted, Override) when
    is_integer(Target)
->
    Parsed = maps:map(fun(_, Texts) -> equation(Texts) end, Equations),
    Families = families([Type], #{}, Schema, Parsed),
    Plain = maps:from_list([
        {T, Minimum(T)}
     || Cs <- maps:values(Families),
        C <- Cs,
        {I, {_, T}} <- lists:enumerate(0, maps:get(fields, C)),
        not lists:member(I, positions(C, Parsed))
    ]),
    Key = {Type, Schema, Equations, Plain},
    Limit = max(0, Target) + 16,
    Cache =
        case get({?MODULE, tables}) of
            undefined -> #{};
            Existing -> Existing
        end,
    {Costs, Choices} =
        case maps:find(Key, Cache) of
            {ok, {Previous, Found, Solutions}} when Previous >= Limit -> {Found, Solutions};
            _ ->
                Found = reach(Families, Parsed, Plain, Limit, #{}),
                Solutions = solutions(Families, Parsed, Plain, Limit, Found),
                put({?MODULE, tables}, Cache#{Key => {Limit, Found, Solutions}}),
                {Found, Solutions}
        end,
    Levels =
        case Target < 0 of
            true -> lists:sublist(lists:sort([K || {T, K} <- maps:keys(Costs), T =:= Type]), 6);
            false -> [Target || maps:is_key({Type, Target}, Costs)]
        end,
    case Levels of
        [] when Target < 0 -> erlang:error({lawspec, {no_indexed_generator, Type}});
        [] ->
            F:exactly('$lawspec_empty_domain');
        _ ->
            F:bind(F:oneof([F:exactly(K) || K <- Levels]), fun(K) ->
                Available = max(Budget, maps:get({Type, K}, Costs)),
                indexed(F, Type, K, Available, Choices, Build, Accepted, Override)
            end)
    end.

indexed(F, Type, K, Budget, Solutions, Build, Accepted, Override) ->
    case Override(Type, K, Budget) of
        {custom, Generator} -> Generator;
        none -> indexed_native(F, Type, K, Budget, Solutions, Build, Accepted, Override)
    end.

indexed_native(F, Type, K, Budget, Solutions, Build, Accepted, Override) ->
    Choices = [
        {C, Targets, Costs}
     || {C, Targets, Costs} <- maps:get({Type, K}, Solutions),
        1 + length(maps:get(witness_values, C, [])) + lists:sum(Costs) =< Budget
    ],
    F:constrain(
        F:bind(
            F:oneof([F:exactly(Choice) || Choice <- Choices]),
            fun({C, Targets, Costs}) ->
                Witnesses = maps:get(witness_values, C, []),
                Count = length(Costs),
                Extra = Budget - 1 - length(Witnesses) - lists:sum(Costs),
                Children = [
                    begin
                        Share =
                            Cost + Extra div Count +
                                case I < Extra rem Count of
                                    true -> 1;
                                    false -> 0
                                end,
                        case maps:find(I, Targets) of
                            {ok, Index} ->
                                indexed(F, T, Index, Share, Solutions, Build, Accepted, Override);
                            error ->
                                Build(T, Share)
                        end
                    end
                 || {I, {{_, T}, Cost}} <- lists:enumerate(0, lists:zip(maps:get(fields, C), Costs))
                ],
                F:bind(F:fixed_list(Children), fun(Values) ->
                    {ls_data, maps:get(tag, C), Values ++ Witnesses}
                end)
            end
        ),
        fun(V) -> Accepted(Type, V) end
    ).

families([], Found, _, _) ->
    Found;
families([T | Pending], Found, S, Equations) ->
    case maps:is_key(T, Found) of
        true ->
            families(Pending, Found, S, Equations);
        false ->
            Cs =
                case lawspec_beam_schema:constructors(T, S) of
                    none ->
                        erlang:error({lawspec, {indexed_type_requires_data, T}});
                    Constructors ->
                        lists:append([lawspec_beam_schema:witness_instances(C) || C <- Constructors])
                end,
            Children = [
                element(2, lists:nth(I + 1, maps:get(fields, C)))
             || C <- Cs,
                I <- positions(C, Equations)
            ],
            families(Children ++ Pending, Found#{T => Cs}, S, Equations)
    end.

positions(C, Equations) -> element(3, maps:get(maps:get(tag, C), Equations)).

%% The finite forward fixpoint supplies the least cost for each reachable
%% index. Positive node costs make cycles harmless; later passes may find a
%% cheaper construction and its parents then inherit that improvement.
reach(Families, Equations, Plain, Limit, Costs) ->
    Next = maps:fold(
        fun(T, Cs, Acc) ->
            lists:foldl(
                fun(C, Current) ->
                    lists:foldl(
                        fun({K, _, Fields}, Table) ->
                            Cost = 1 + length(maps:get(witness_values, C, [])) + lists:sum(Fields),
                            Key = {T, K},
                            case maps:find(Key, Table) of
                                {ok, Existing} when Existing =< Cost -> Table;
                                _ -> Table#{Key => Cost}
                            end
                        end,
                        Current,
                        assignments(C, Equations, Plain, Limit, Costs)
                    )
                end,
                Acc,
                Cs
            )
        end,
        Costs,
        Families
    ),
    case Next =:= Costs of
        true -> Costs;
        false -> reach(Families, Equations, Plain, Limit, Next)
    end.

solutions(Families, Equations, Plain, Limit, Costs) ->
    maps:fold(
        fun(T, Cs, Table) ->
            lists:foldl(
                fun(C, Acc) ->
                    lists:foldl(
                        fun({K, Targets, Fields}, Current) ->
                            Key = {T, K},
                            Current#{Key => maps:get(Key, Current, []) ++ [{C, Targets, Fields}]}
                        end,
                        Acc,
                        assignments(C, Equations, Plain, Limit, Costs)
                    )
                end,
                Table,
                Cs
            )
        end,
        #{},
        Families
    ).

assignments(C, Equations, Plain, Limit, Costs) ->
    {Term, Guards, Positions} = maps:get(maps:get(tag, C), Equations),
    Fields = maps:get(fields, C),
    case
        lists:all(
            fun({I, {_, T}}) -> lists:member(I, Positions) orelse maps:get(T, Plain) =/= none end,
            lists:enumerate(0, Fields)
        )
    of
        false ->
            [];
        true ->
            Choices = [
                {I, lists:sort([K || {T0, K} <- maps:keys(Costs), T0 =:= T])}
             || I <- Positions,
                {_, T} <- [lists:nth(I + 1, Fields)]
            ],
            [
                begin
                    Required = [
                        case maps:find(I, Assigned) of
                            {ok, K} -> maps:get({T, K}, Costs);
                            error -> maps:get(T, Plain)
                        end
                     || {I, {_, T}} <- lists:enumerate(0, Fields)
                    ],
                    {Value, Assigned, Required}
                end
             || Assigned <- assign(Choices, Guards, #{}),
                Value <- [evaluate(Term, Assigned)],
                is_integer(Value),
                Value >= 0,
                Value =< Limit
            ]
    end.

assign(Choices, Guards, Assigned) ->
    Ready = [
        G
     || G = {_, A, B} <- Guards,
        lists:all(fun(I) -> maps:is_key(I, Assigned) end, fields(A) ++ fields(B))
    ],
    case lists:all(fun(G) -> holds(G, Assigned) end, Ready) of
        false ->
            [];
        true ->
            case Choices of
                [] ->
                    [Assigned];
                [{I, Values} | Rest] ->
                    lists:append([assign(Rest, Guards, Assigned#{I => V}) || V <- Values])
            end
    end.

equation([Text | Guards]) ->
    {Term, []} = parse(tokens(Text)),
    Parsed = [guard(G) || G <- Guards],
    Positions = lists:usort(
        fields(Term) ++ lists:append([fields(A) ++ fields(B) || {_, A, B} <- Parsed])
    ),
    {Term, Parsed, Positions}.

guard(Text) ->
    [Op | Rest] = tokens(Text),
    true = Op =:= <<"==">> orelse Op =:= <<">=">>,
    {A, Next} = parse(Rest),
    {B, []} = parse(Next),
    {Op, A, B}.

tokens(Text) -> binary:split(Text, <<" ">>, [global, trim_all]).
parse([<<"c", N/binary>> | Rest]) ->
    {{constant, binary_to_integer(N)}, Rest};
parse([<<"f", N/binary>> | Rest]) ->
    {{field, binary_to_integer(N)}, Rest};
parse([Op | Rest]) ->
    true = lists:member(Op, [<<"+">>, <<"-">>, <<"*">>, <<"div">>, <<"mod">>, <<"^">>]),
    {A, Next} = parse(Rest),
    {B, After} = parse(Next),
    {{Op, A, B}, After}.

fields({constant, _}) -> [];
fields({field, I}) -> [I];
fields({_, A, B}) -> fields(A) ++ fields(B).

evaluate({constant, N}, _) -> N;
evaluate({field, I}, Assigned) -> maps:get(I, Assigned, undefined);
evaluate({Op, A, B}, Assigned) -> binary(Op, evaluate(A, Assigned), evaluate(B, Assigned)).

binary(_, undefined, _) -> undefined;
binary(_, _, undefined) -> undefined;
binary(<<"+">>, A, B) -> A + B;
binary(<<"-">>, A, B) when A >= B -> A - B;
binary(<<"*">>, A, B) -> A * B;
binary(<<"div">>, A, B) when B > 0 -> A div B;
binary(<<"mod">>, A, B) when B > 0 -> A rem B;
binary(<<"^">>, A, B) when B >= 0, B =< 64 -> power(A, B);
binary(_, _, _) -> undefined.
power(_, 0) -> 1;
power(A, B) when B rem 2 =:= 0 -> power(A * A, B div 2);
power(A, B) -> A * power(A, B - 1).

holds({Relation, A, B}, Assigned) ->
    X = evaluate(A, Assigned),
    Y = evaluate(B, Assigned),
    is_integer(X) andalso is_integer(Y) andalso
        case Relation of
            <<"==">> -> X =:= Y;
            <<">=">> -> X >= Y
        end.

with_cache(Run) ->
    Previous = erase({?MODULE, tables}),
    try
        Run()
    after
        case Previous of
            undefined -> erase({?MODULE, tables});
            _ -> put({?MODULE, tables}, Previous)
        end
    end.
