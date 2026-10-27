%% @doc Model and scenario descriptors share seeded generation, shrinking,
%% rendering and canonical wire bytes with every other target. Descriptor
%% identifiers remain binaries: reading input never interns a BEAM atom.
%% ref:DEC-portable-seeded-generation ref:DEC-distribution-canonical-wire
-module(lawspec_beam_values).
-export([
    read_descriptor/1,
    from_text/1,
    generate/4,
    minimal/2,
    shrink/3,
    render/1,
    encode/3,
    decode/3,
    decode_prefix/3,
    resolve/2,
    varint/1,
    read_varint/1
]).

read_descriptor(Text) when is_binary(Text) ->
    {Forms, Rest} = forms(unicode:characters_to_list(Text), []),
    require(Rest =:= [], trailing_descriptor_input),
    Forms.
forms(Cs, Acc) ->
    case skip(Cs) of
        [] ->
            {lists:reverse(Acc), []};
        [$) | _] = Rest ->
            {lists:reverse(Acc), Rest};
        Rest ->
            {V, Next} = item(Rest),
            forms(Next, [V | Acc])
    end.
skip([C | Cs]) when C =:= $\s; C =:= $\n; C =:= $\r; C =:= $\t -> skip(Cs);
skip(Cs) -> Cs.
item([$( | Cs]) ->
    {Children, Rest} = forms(Cs, []),
    case Rest of
        [$) | Next] -> {Children, Next};
        _ -> fail(unclosed_descriptor)
    end;
item([$" | Cs]) ->
    quoted(Cs, []);
item(Cs) ->
    {Chars, Rest} = lists:splitwith(fun(C) -> not lists:member(C, " \t\r\n()") end, Cs),
    require(Chars =/= [], empty_descriptor_atom),
    Value =
        case Chars of
            "_" ->
                none;
            _ ->
                case string:to_integer(Chars) of
                    {N, []} -> N;
                    _ -> unicode:characters_to_binary(Chars)
                end
        end,
    {Value, Rest}.
quoted([$\\, C | Cs], Acc) -> quoted(Cs, [C | Acc]);
quoted([$" | Cs], Acc) -> {{quoted, unicode:characters_to_binary(lists:reverse(Acc))}, Cs};
quoted([C | Cs], Acc) -> quoted(Cs, [C | Acc]);
quoted([], _) -> fail(unclosed_descriptor_string).

from_text(Text) ->
    Forms = read_descriptor(Text),
    require(Forms =/= [], empty_descriptor),
    {
        maps:from_list([{unquote(Name), D} || [<<"data">>, Name | _] = D <- Forms]),
        lists:last(Forms)
    }.
unquote({quoted, B}) -> B;
unquote(B) -> B.
resolve([<<"ref">>, Name], Table) -> maps:get(unquote(Name), Table);
resolve(D, _) -> D.

%% @doc These draws intentionally follow the portable descriptor algorithm,
%% including each boundary-choice draw, so a seed replays an existing run.
%% ref:DEC-portable-seeded-generation
generate(D, Table, State, Size) when is_integer(Size), Size >= 0 ->
    generate_resolved(resolve(D, Table), Table, State, Size).
generate_resolved([<<"int">>, _, _, _] = D, _, State, _) ->
    {Lo, Hi} = generation_bounds(D),
    {Choice, S1} = lawspec_beam_random:below(10, State),
    case Choice < 2 of
        true ->
            {I, S2} = lawspec_beam_random:below(4, S1),
            {lists:nth(I + 1, [Lo, Hi, min(max(0, Lo), Hi), min(max(1, Lo), Hi)]), S2};
        false ->
            {N, S2} = lawspec_beam_random:below(Hi - Lo + 1, S1),
            {Lo + N, S2}
    end;
generate_resolved([<<"bool">>], _, State, _) ->
    {N, Next} = lawspec_beam_random:below(2, State),
    {N =:= 1, Next};
generate_resolved([<<"text">>], _, State, Size) ->
    {Count, S1} = lawspec_beam_random:below(Size + 1, State),
    {Chars, S2} = draw_n(
        Count,
        fun(S) ->
            {N, S0} = lawspec_beam_random:below(95, S),
            {N + 32, S0}
        end,
        S1
    ),
    {list_to_binary(Chars), S2};
generate_resolved([<<"unit">>], _, State, _) ->
    {ls_unit, State};
generate_resolved([<<"list">>, D], Table, State, Size) ->
    {Count, S1} = lawspec_beam_random:below(Size + 1, State),
    draw_n(Count, fun(S) -> generate(D, Table, S, Size) end, S1);
generate_resolved([<<"maybe">>, D], Table, State, Size) ->
    {Choice, S1} = lawspec_beam_random:below(4, State),
    case Choice of
        0 ->
            {{ls_data, <<"Maybe::Absent">>, []}, S1};
        _ ->
            {V, S2} = generate(D, Table, S1, Size),
            {{ls_data, <<"Maybe::Present">>, [V]}, S2}
    end;
generate_resolved([<<"either">>, A, B], Table, State, Size) ->
    {Choice, S1} = lawspec_beam_random:below(2, State),
    {Tag, D} =
        case Choice of
            0 -> {<<"Either::Left">>, A};
            _ -> {<<"Either::Right">>, B}
        end,
    {V, S2} = generate(D, Table, S1, Size),
    {{ls_data, Tag, [V]}, S2};
generate_resolved([<<"data">>, _ | Cs], Table, State, Size) ->
    Choices =
        case Size of
            0 -> base(Cs);
            _ -> Cs
        end,
    {I, S1} = lawspec_beam_random:below(length(Choices), State),
    [<<"ctor">>, Tag | Ds] = lists:nth(I + 1, Choices),
    {Fields, S2} = lists:mapfoldl(fun(D, S) -> generate(D, Table, S, max(Size - 1, 0)) end, S1, Ds),
    {{ls_data, unquote(Tag), Fields}, S2};
generate_resolved(D, _, _, _) ->
    fail({unsupported_generation_descriptor, D}).

draw_n(N, F, State) -> lists:mapfoldl(fun(_, S) -> F(S) end, State, lists:seq(1, N)).
generation_bounds([<<"int">>, _, none, none]) -> {-1000000, 1000000};
generation_bounds([<<"int">>, _, none, Hi]) -> {min(-1000000, Hi - 2000000), Hi};
generation_bounds([<<"int">>, _, Lo, none]) -> {Lo, max(1000000, Lo + 2000000)};
generation_bounds([<<"int">>, _, Lo, Hi]) -> {Lo, Hi}.
base(Cs) ->
    Found = [C || [<<"ctor">>, _ | Fields] = C <- Cs, not lists:any(fun mentions_data/1, Fields)],
    case Found of
        [] -> Cs;
        _ -> Found
    end.
mentions_data([<<"ref">> | _]) -> true;
mentions_data([<<"data">> | _]) -> true;
mentions_data([_ | Fields]) -> lists:any(fun mentions_data/1, Fields);
mentions_data(_) -> false.

minimal(D, Table) -> minimal_resolved(resolve(D, Table), Table).
minimal_resolved([<<"int">>, _, _, _] = D, _) ->
    {Lo, Hi} = generation_bounds(D),
    min(max(0, Lo), Hi);
minimal_resolved([<<"bool">>], _) ->
    false;
minimal_resolved([<<"text">>], _) ->
    <<>>;
minimal_resolved([<<"unit">>], _) ->
    ls_unit;
minimal_resolved([<<"list">>, _], _) ->
    [];
minimal_resolved([<<"maybe">>, _], _) ->
    {ls_data, <<"Maybe::Absent">>, []};
minimal_resolved([<<"either">>, A, _], Table) ->
    {ls_data, <<"Either::Left">>, [minimal(A, Table)]};
minimal_resolved([<<"data">>, _ | Cs], Table) ->
    [[<<"ctor">>, Tag | Ds] | _] = base(Cs),
    {ls_data, unquote(Tag), [minimal(D, Table) || D <- Ds]}.

%% @doc Shrinks are ordered and deduplicated identically to the portable
%% runtime, and integer candidates remain inside the descriptor's bounds.
%% ref:DEC-shrink-within-domain
shrink(D, V, Table) ->
    Candidates = shrink_resolved(resolve(D, Table), V, Table),
    unique(Candidates, [V], []).
unique([], _, Acc) ->
    lists:reverse(Acc);
unique([C | Cs], Seen, Acc) ->
    case lists:member(C, Seen) of
        true -> unique(Cs, Seen, Acc);
        false -> unique(Cs, [C | Seen], [C | Acc])
    end.
shrink_resolved([<<"int">>, _, _, _] = D, V, Table) ->
    Target = minimal(D, Table),
    case V =:= Target of
        true ->
            [];
        false ->
            [
                Target,
                V - (V - Target) div 2,
                V -
                    case V > Target of
                        true -> 1;
                        false -> -1
                    end
            ]
    end;
shrink_resolved([<<"bool">>], true, _) ->
    [false];
shrink_resolved([<<"bool">>], false, _) ->
    [];
shrink_resolved([<<"text">>], V, _) ->
    [unicode:characters_to_binary(Cs) || Cs <- shrink_sequence(unicode:characters_to_list(V))];
shrink_resolved([<<"list">>, D], Vs, Table) ->
    shrink_sequence(Vs) ++ replace_shrinks(Vs, fun(V) -> shrink(D, V, Table) end);
shrink_resolved([<<"maybe">>, D], {ls_data, <<"Maybe::Present">>, [V]}, Table) ->
    [
        {ls_data, <<"Maybe::Absent">>, []}
        | [{ls_data, <<"Maybe::Present">>, [C]} || C <- shrink(D, V, Table)]
    ];
shrink_resolved([<<"maybe">>, _], {ls_data, <<"Maybe::Absent">>, []}, _) ->
    [];
shrink_resolved([<<"either">>, A, B], {ls_data, Tag, [V]}, Table) ->
    D =
        case Tag of
            <<"Either::Left">> -> A;
            <<"Either::Right">> -> B
        end,
    [{ls_data, Tag, [C]} || C <- shrink(D, V, Table)];
shrink_resolved([<<"data">>, Name | Cs] = D, {ls_data, Tag, Fields}, Table) ->
    [<<"ctor">>, _ | Ds] = constructor(Tag, Cs),
    [minimal(D, Table)] ++ [F || {F, FD} <- lists:zip(Fields, Ds), FD =:= [<<"ref">>, Name]] ++
        [{ls_data, Tag, Fs} || Fs <- field_shrinks(Fields, Ds, Table)];
shrink_resolved([<<"unit">>], ls_unit, _) ->
    [].
shrink_sequence([]) ->
    [];
shrink_sequence(Vs) ->
    [[], lists:sublist(Vs, length(Vs) div 2)] ++
        [A ++ Rest || I <- lists:seq(0, length(Vs) - 1), {A, [_ | Rest]} <- [lists:split(I, Vs)]].
replace_shrinks(Vs, Shrink) ->
    [
        A ++ [C | Rest]
     || I <- lists:seq(0, length(Vs) - 1),
        {A, [V | Rest]} <- [lists:split(I, Vs)],
        C <- Shrink(V)
    ].
field_shrinks(Vs, Ds, Table) ->
    [
        A ++ [C | Rest]
     || {I, D} <- lists:zip(lists:seq(0, length(Ds) - 1), Ds),
        {A, [V | Rest]} <- [lists:split(I, Vs)],
        C <- shrink(D, V, Table)
    ].

render(V) -> iolist_to_binary(render_value(V)).
render_value(true) ->
    <<"true">>;
render_value(false) ->
    <<"false">>;
render_value(ls_unit) ->
    <<"()">>;
render_value(N) when is_integer(N) -> integer_to_binary(N);
render_value(B) when is_binary(B) ->
    Escaped = binary:replace(
        binary:replace(B, <<"\\">>, <<"\\\\">>, [global]), <<"\"">>, <<"\\\"">>, [global]
    ),
    [$", Escaped, $"];
render_value(Vs) when is_list(Vs) -> [$[, lists:join(<<", ">>, [render_value(V) || V <- Vs]), $]];
render_value({ls_data, Tag, Fields}) ->
    Name = lists:last(binary:split(Tag, <<"::">>, [global])),
    case Fields of
        [] -> Name;
        _ -> [Name, $(, lists:join(<<", ">>, [render_value(V) || V <- Fields]), $)]
    end.

encode(D, V, Table) -> iolist_to_binary(put(resolve(D, Table), V, Table)).
put([<<"int">>, _, Lo, Hi], V, _) ->
    check_integer(V, Lo, Hi),
    varint(
        case V >= 0 of
            true -> V * 2;
            false -> -V * 2 - 1
        end
    );
put([<<"bool">>], V, _) when is_boolean(V) ->
    case V of
        true -> <<1>>;
        false -> <<0>>
    end;
put([<<"unit">>], ls_unit, _) ->
    <<>>;
put([Kind], V, _) when Kind =:= <<"text">>; Kind =:= <<"bytes">>; Kind =:= <<"end">> ->
    check_binary(Kind, V),
    [varint(byte_size(V)), V];
put([<<"list">>, D], Vs, Table) when is_list(Vs) ->
    [varint(length(Vs)), [put(resolve(D, Table), V, Table) || V <- Vs]];
put([<<"maybe">>, _], {ls_data, <<"Maybe::Absent">>, []}, _) ->
    <<0>>;
put([<<"maybe">>, D], {ls_data, <<"Maybe::Present">>, [V]}, Table) ->
    [1, put(resolve(D, Table), V, Table)];
put([<<"either">>, A, _], {ls_data, <<"Either::Left">>, [V]}, Table) ->
    [0, put(resolve(A, Table), V, Table)];
put([<<"either">>, _, B], {ls_data, <<"Either::Right">>, [V]}, Table) ->
    [1, put(resolve(B, Table), V, Table)];
put([<<"data">>, _ | Cs], {ls_data, Tag, Fields}, Table) ->
    C = constructor(Tag, Cs),
    [<<"ctor">>, _ | Ds] = C,
    require(length(Ds) =:= length(Fields), wire_field_count),
    Index = length(lists:takewhile(fun(Other) -> Other =/= C end, Cs)),
    [varint(Index), [put(resolve(D, Table), V, Table) || {D, V} <- lists:zip(Ds, Fields)]];
put(D, _, _) ->
    fail({invalid_wire_value, D}).
constructor(Tag, Cs) ->
    case [C || [<<"ctor">>, T | _] = C <- Cs, unquote(T) =:= Tag] of
        [C] -> C;
        _ -> fail({foreign_wire_constructor, Tag})
    end.
check_integer(V, Lo, Hi) ->
    require(
        is_integer(V) andalso (Lo =:= none orelse V >= Lo) andalso
            (Hi =:= none orelse V =< Hi),
        wire_integer_out_of_range
    ).
check_binary(Kind, V) ->
    require(is_binary(V), wire_binary_required),
    case Kind of
        <<"bytes">> -> ok;
        _ -> require(is_list(unicode:characters_to_list(V)), wire_invalid_utf8)
    end.

varint(N) when is_integer(N), N >= 0, N < 128 -> <<N>>;
varint(N) when is_integer(N), N >= 128 -> [(N band 127) bor 128 | varint_list(N bsr 7)].
varint_list(N) when N < 128 -> [N];
varint_list(N) -> [(N band 127) bor 128 | varint_list(N bsr 7)].
read_varint(B) -> read_varint(B, 0, 0).
read_varint(<<Byte, Rest/binary>>, Acc, Shift) ->
    N = Acc bor ((Byte band 127) bsl Shift),
    case Byte < 128 of
        true -> {N, Rest};
        false -> read_varint(Rest, N, Shift + 7)
    end;
read_varint(<<>>, _, _) ->
    fail(truncated_wire_value).

decode(D, Bytes, Table) ->
    {V, Rest} = decode_prefix(D, Bytes, Table),
    require(Rest =:= <<>>, trailing_wire_bytes),
    V.
decode_prefix(D, Bytes, Table) -> get(resolve(D, Table), Bytes, Table).
get([<<"int">>, _, Lo, Hi], B, _) ->
    {Z, Rest} = read_varint(B),
    V =
        case Z rem 2 of
            0 -> Z div 2;
            1 -> -(Z + 1) div 2
        end,
    check_integer(V, Lo, Hi),
    {V, Rest};
get([<<"bool">>], <<N, Rest/binary>>, _) when N < 2 -> {N =:= 1, Rest};
get([<<"unit">>], B, _) ->
    {ls_unit, B};
get([Kind], B, _) when Kind =:= <<"text">>; Kind =:= <<"bytes">>; Kind =:= <<"end">> ->
    {N, Rest} = read_varint(B),
    require(byte_size(Rest) >= N, truncated_wire_value),
    <<V:N/binary, After/binary>> = Rest,
    check_binary(Kind, V),
    {V, After};
get([<<"list">>, D], B, Table) ->
    {N, Rest} = read_varint(B),
    get_fields(lists:duplicate(N, D), Rest, Table);
get([<<"maybe">>, _], <<0, Rest/binary>>, _) ->
    {{ls_data, <<"Maybe::Absent">>, []}, Rest};
get([<<"maybe">>, D], <<1, Rest/binary>>, Table) ->
    {V, After} = get(resolve(D, Table), Rest, Table),
    {{ls_data, <<"Maybe::Present">>, [V]}, After};
get([<<"either">>, A, B], <<N, Rest/binary>>, Table) when N < 2 ->
    {Tag, D} =
        case N of
            0 -> {<<"Either::Left">>, A};
            1 -> {<<"Either::Right">>, B}
        end,
    {V, After} = get(resolve(D, Table), Rest, Table),
    {{ls_data, Tag, [V]}, After};
get([<<"data">>, _ | Cs], B, Table) ->
    {I, Rest} = read_varint(B),
    require(I < length(Cs), invalid_wire_constructor),
    [<<"ctor">>, Tag | Ds] = lists:nth(I + 1, Cs),
    {Fields, After} = get_fields(Ds, Rest, Table),
    {{ls_data, unquote(Tag), Fields}, After};
get(D, _, _) ->
    fail({invalid_wire_value, D}).
get_fields(Ds, B, Table) ->
    lists:mapfoldl(fun(D, Rest) -> get(resolve(D, Table), Rest, Table) end, B, Ds).

require(true, _) -> ok;
require(false, Reason) -> fail(Reason).
fail(Reason) -> erlang:error({lawspec, Reason}).
