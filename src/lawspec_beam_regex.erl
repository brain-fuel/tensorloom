%% @doc The portable regex grammar runs on Unicode scalar values, not bytes
%% or the host regex library's character classes. Matching covers the whole
%% text. Reachable positions avoid backtracking through ambiguous repeats.
%% ref:DEC-typed-core-boundary
-module(lawspec_beam_regex).
-export([compile/1, matches/2, run/2]).

compile(Pattern) ->
    _ = lawspec_beam_scalar:validate(Pattern, <<"Text">>, 64),
    {Node, Rest} = alternatives(unicode:characters_to_list(Pattern)),
    require(Rest =:= [], unmatched_closing_parenthesis),
    Node.

matches(Pattern, Text) -> run(compile(Pattern), Text).
run(Node, Text) ->
    _ = lawspec_beam_scalar:validate(Text, <<"Text">>, 64),
    Points = list_to_tuple(unicode:characters_to_list(Text)),
    lists:member(tuple_size(Points), reach(Node, Points, [0])).

alternatives(Cs) ->
    {Branch, Rest} = sequence(Cs, []),
    alternatives(Rest, [Branch]).
alternatives([$| | Rest], Acc) ->
    {Branch, Next} = sequence(Rest, []),
    alternatives(Next, [Branch | Acc]);
alternatives(Rest, [Only]) ->
    {Only, Rest};
alternatives(Rest, Acc) ->
    {{alt, lists:reverse(Acc)}, Rest}.

sequence([], Acc) ->
    {{seq, lists:reverse(Acc)}, []};
sequence([C | _] = Rest, Acc) when C =:= $); C =:= $| ->
    {{seq, lists:reverse(Acc)}, Rest};
sequence(Cs, Acc) ->
    {Atom, Rest} = atom(Cs),
    {Node, Next} = quantified(Atom, Rest),
    sequence(Next, [Node | Acc]).

atom([$(, $?, $: | Cs]) ->
    group(Cs);
atom([$(, $? | _]) ->
    fail(unsupported_group);
atom([$( | Cs]) ->
    group(Cs);
atom([$[, $^ | Cs]) ->
    char_class(Cs, true, []);
atom([$[ | Cs]) ->
    char_class(Cs, false, []);
atom([$. | Cs]) ->
    {{set, true, [{false, [{10, 10}]}]}, Cs};
atom([$\\ | Cs]) ->
    {Item, Rest} = escape(Cs),
    {{set, false, [Item]}, Rest};
atom([C | Cs]) ->
    require(not lists:member(C, "*+?{^$]}"), {unexpected_character, C}),
    {{set, false, [{false, [{C, C}]}]}, Cs};
atom([]) ->
    fail(missing_atom).
group(Cs) ->
    {Node, Rest} = alternatives(Cs),
    case Rest of
        [$) | Next] -> {Node, Next};
        _ -> fail(unclosed_group)
    end.

escape([C | Cs]) when C =:= $d; C =:= $D -> {{C =:= $D, [{48, 57}]}, Cs};
escape([C | Cs]) when C =:= $w; C =:= $W ->
    {{C =:= $W, [{48, 57}, {65, 90}, {95, 95}, {97, 122}]}, Cs};
escape([C | Cs]) when C =:= $s; C =:= $S -> {{C =:= $S, [{9, 13}, {32, 32}]}, Cs};
escape([C | Cs]) ->
    case lists:keyfind(C, 1, [{$n, 10}, {$t, 9}, {$r, 13}, {$f, 12}, {$v, 11}]) of
        {_, V} ->
            {{false, [{V, V}]}, Cs};
        false ->
            require(lists:member(C, "\\.^$|?*+()[]{}-/"), {unsupported_escape, C}),
            {{false, [{C, C}]}, Cs}
    end;
escape([]) ->
    fail(trailing_escape).

class_item([$\\ | Cs]) -> escape(Cs);
class_item([$[ | _]) -> fail(unescaped_class_open);
class_item([C | Cs]) -> {{false, [{C, C}]}, Cs};
class_item([]) -> fail(unclosed_class).
char_class([], _, _) ->
    fail(unclosed_class);
char_class([$] | _], _, []) ->
    fail(empty_class);
char_class([$] | Cs], Negated, Acc) ->
    {{set, Negated, lists:reverse(Acc)}, Cs};
char_class(Cs, Negated, Acc) ->
    {Item, Rest} = class_item(Cs),
    case {Item, Rest} of
        {{false, [{Lo, Lo}]}, [$-, C | Tail]} when C =/= $] ->
            {High, Next} = class_item([C | Tail]),
            case High of
                {false, [{Hi, Hi}]} when Hi >= Lo ->
                    char_class(Next, Negated, [{false, [{Lo, Hi}]} | Acc]);
                _ ->
                    fail(invalid_class_range)
            end;
        _ ->
            char_class(Rest, Negated, [Item | Acc])
    end.

quantified(Node, [$* | Cs]) ->
    repeat(Node, 0, infinity, Cs);
quantified(Node, [$+ | Cs]) ->
    repeat(Node, 1, infinity, Cs);
quantified(Node, [$? | Cs]) ->
    repeat(Node, 0, 1, Cs);
quantified(Node, [${ | Cs]) ->
    {Low, Rest} = count(Cs),
    case Rest of
        [$} | Next] ->
            repeat(Node, Low, Low, Next);
        [$,, $} | Next] ->
            repeat(Node, Low, infinity, Next);
        [$, | Tail] ->
            {High, After} = count(Tail),
            case After of
                [$} | Next] -> repeat(Node, Low, High, Next);
                _ -> fail(unclosed_repetition)
            end;
        _ ->
            fail(invalid_repetition)
    end;
quantified(Node, Cs) ->
    {Node, Cs}.

repeat(Node, Low, High, Cs) ->
    require(
        Low =< 1000 andalso (High =:= infinity orelse (High =< 1000 andalso High >= Low)),
        invalid_repetition_count
    ),
    case Cs of
        [C | _] -> require(not lists:member(C, "*+?{"), repeated_quantifier);
        [] -> ok
    end,
    {{repeat, Node, Low, High}, Cs}.
count(Cs) ->
    {Digits, Rest} = lists:splitwith(fun(C) -> C >= $0 andalso C =< $9 end, Cs),
    require(Digits =/= [], missing_repetition_count),
    {list_to_integer(Digits), Rest}.

reach({set, Negated, Items}, Text, Positions) ->
    [
        P + 1
     || P <- Positions,
        P < tuple_size(Text),
        in_set(element(P + 1, Text), Items) =/= Negated
    ];
reach({seq, Nodes}, Text, Positions) ->
    lists:foldl(fun(Node, Ps) -> reach(Node, Text, Ps) end, Positions, Nodes);
reach({alt, Nodes}, Text, Positions) ->
    ordsets:union([reach(Node, Text, Positions) || Node <- Nodes]);
reach({repeat, Node, Low, High}, Text, Positions) ->
    Required = repeat_required(Node, Text, Positions, Low),
    Limit =
        case High of
            infinity -> infinity;
            _ -> High - Low
        end,
    repeat_reach(Node, Text, Required, Required, Limit).
in_set(C, Items) ->
    lists:any(
        fun({Neg, Ranges}) ->
            Neg =/= lists:any(fun({Lo, Hi}) -> C >= Lo andalso C =< Hi end, Ranges)
        end,
        Items
    ).
repeat_required(_, _, [], _) -> [];
repeat_required(_, _, Ps, 0) -> Ps;
repeat_required(Node, Text, Ps, N) -> repeat_required(Node, Text, reach(Node, Text, Ps), N - 1).
repeat_reach(_, _, _, Seen, 0) ->
    Seen;
repeat_reach(_, _, [], Seen, _) ->
    Seen;
repeat_reach(Node, Text, Frontier, Seen, Limit) ->
    Next = ordsets:subtract(reach(Node, Text, Frontier), Seen),
    Remaining =
        case Limit of
            infinity -> infinity;
            _ -> Limit - 1
        end,
    repeat_reach(Node, Text, Next, ordsets:union(Seen, Next), Remaining).

require(true, _) -> ok;
require(false, Reason) -> fail(Reason).
fail(Reason) -> erlang:error({lawspec, {non_portable_regex, Reason}}).
