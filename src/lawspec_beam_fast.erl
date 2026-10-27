%% @doc Typed native code's runtime: checked narrowing, exact arithmetic,
%% sequences by code point, KeyVal and Set as maps in the portable order, and
%% the state of a typed spec handler, which stays in its cell's process: an
%% operation runs there, so only its arguments and its result are copied.
%% ref:DEC-typed-native-definitions
-module(lawspec_beam_fast).
-export([
    int8/1,
    int16/1,
    int32/1,
    int64/1,
    uint8/1,
    uint16/1,
    uint32/1,
    uint64/1,
    quot/2,
    remainder/2,
    power/2,
    unreachable/0,
    discard/1,
    text_length/1,
    starts_with/2,
    ends_with/2,
    contains/2,
    slice_text/3,
    slice_bytes/3,
    slice_list/3,
    at_text/2,
    at_bytes/2,
    at_list/2,
    range/2,
    join/2,
    compare/2,
    sorted_pairs/1, sorted_pairs/2,
    sorted_items/1, sorted_items/2,
    set_of/1,
    key_val_of/1,
    handler/3,
    update/2,
    update_tracked/2,
    state/1,
    set_state/2
]).

%% Bounded integers: a result outside its type fails as the dynamic code does.
int8(V) when V >= -128, V =< 127 -> V;
int8(_) -> outside(<<"Int8">>).
int16(V) when V >= -32768, V =< 32767 -> V;
int16(_) -> outside(<<"Int16">>).
int32(V) when V >= -2147483648, V =< 2147483647 -> V;
int32(_) -> outside(<<"Int32">>).
int64(V) when V >= -9223372036854775808, V =< 9223372036854775807 -> V;
int64(_) -> outside(<<"Int64">>).
uint8(V) when V >= 0, V =< 255 -> V;
uint8(_) -> outside(<<"UInt8">>).
uint16(V) when V >= 0, V =< 65535 -> V;
uint16(_) -> outside(<<"UInt16">>).
uint32(V) when V >= 0, V =< 4294967295 -> V;
uint32(_) -> outside(<<"UInt32">>).
uint64(V) when V >= 0, V =< 18446744073709551615 -> V;
uint64(_) -> outside(<<"UInt64">>).
outside(T) -> erlang:error({lawspec, {integer_out_of_range, T}}).

%% Exact division truncates toward zero; its remainder has the dividend's sign.
quot(A, B) when B =/= 0 -> A div B;
quot(_, _) -> erlang:error({lawspec, {invalid_exact_operation, <<"quot">>}}).
remainder(A, B) when B =/= 0 -> A rem B;
remainder(_, _) -> erlang:error({lawspec, {invalid_exact_operation, <<"rem">>}}).
power(_, 0) ->
    1;
power(A, N) when N > 0 ->
    H = power(A, N div 2),
    case N rem 2 of
        0 -> H * H;
        1 -> H * H * A
    end;
power(_, _) ->
    erlang:error({lawspec, {invalid_exact_operation, <<"pow">>}}).

unreachable() -> erlang:error({lawspec, unreachable}).

%% A value computed for its effects alone, as a let whose name is unused.
discard(_) -> ok.

%% Text is UTF-8, indexed by code point: its length counts the bytes that
%% start a code point.
text_length(T) -> text_length(T, 0).
text_length(<<>>, N) -> N;
text_length(<<2#10:2, _:6, Rest/binary>>, N) -> text_length(Rest, N);
text_length(<<_, Rest/binary>>, N) -> text_length(Rest, N + 1).

starts_with(Text, Prefix) ->
    byte_size(Text) >= byte_size(Prefix) andalso binary:part(Text, 0, byte_size(Prefix)) =:= Prefix.
ends_with(Text, Suffix) ->
    byte_size(Text) >= byte_size(Suffix) andalso
        binary:part(Text, byte_size(Text) - byte_size(Suffix), byte_size(Suffix)) =:= Suffix.
contains(Text, Part) -> Part =:= <<>> orelse binary:match(Text, Part) =/= nomatch.

%% A slice is clamped to the sequence.
slice_text(T, From, To) ->
    Points = unicode:characters_to_list(T),
    {I, J} = bounds(From, To, length(Points)),
    unicode:characters_to_binary(lists:sublist(Points, I + 1, J - I)).
slice_bytes(B, From, To) ->
    {I, J} = bounds(From, To, byte_size(B)),
    binary:part(B, I, J - I).
slice_list(Xs, From, To) ->
    {I, J} = bounds(From, To, length(Xs)),
    lists:sublist(lists:nthtail(I, Xs), J - I).
bounds(From, To, Count) ->
    I = clamp(From, 0, Count),
    {I, clamp(To, I, Count)}.
clamp(N, Lo, _) when N < Lo -> Lo;
clamp(N, _, Hi) when N > Hi -> Hi;
clamp(N, _, _) -> N.

%% An item at an index: {ok, Item} or error, made a Maybe by the caller.
at_text(T, I) -> at_list(unicode:characters_to_list(T), I).
at_bytes(B, I) when I >= 0, I < byte_size(B) -> {ok, binary:at(B, I)};
at_bytes(_, _) -> error.
at_list(Xs, I) when I >= 0 -> at_from(Xs, I);
at_list(_, _) -> error.
at_from([X | _], 0) -> {ok, X};
at_from([_ | Rest], I) -> at_from(Rest, I - 1);
at_from([], _) -> error.

range(Lo, Hi) when Hi > Lo -> lists:seq(Lo, Hi - 1);
range(_, _) -> [].

join(Separator, Parts) -> iolist_to_binary(lists:join(Separator, Parts)).

%% The portable order of values whose Erlang term order is it: scalars,
%% lists, and data whose constructors sort as their tags do.
compare(A, B) when A < B -> -1;
compare(A, B) when A > B -> 1;
compare(_, _) -> 0.

%% A KeyVal is a map and a Set a version 2 set; their entries and items are
%% given in the portable order of their keys.
sorted_pairs(M) -> lists:sort(maps:to_list(M)).
sorted_pairs(M, Compare) ->
    lists:sort(fun({A, _}, {B, _}) -> Compare(A, B) =< 0 end, maps:to_list(M)).
sorted_items(S) -> lists:sort(sets:to_list(S)).
sorted_items(S, Compare) -> lists:sort(fun(A, B) -> Compare(A, B) =< 0 end, sets:to_list(S)).
set_of(Items) -> sets:from_list(Items, [{version, 2}]).
%% Later entries win, as prelude.keyValOf puts them in order.
key_val_of(Pairs) -> maps:from_list(Pairs).

%% A typed spec handler with state: a cell whose operations run in its own
%% process. The state never leaves it; an operation that fails leaves it as
%% it was. Without state, the handler is its operations.
handler(Schema, Initial, Operations) ->
    {lawspec_typed, lawspec_beam_effects:typed_cell(Schema, Initial), Operations}.

%% Run Step over the state: Step(State) gives {Result, NextState}.
update({lawspec_typed, Cell, _}, Step) -> lawspec_beam_handler:in_place(Cell, Step).

%% The same for an operation that performs another ability's operations,
%% which joins the handlers' dependency graph.
update_tracked({lawspec_typed, Cell, _}, Step) -> lawspec_beam_handler:in_place_tracked(Cell, Step).

%% The typed state: a snapshot, copied out of the cell's process.
state({lawspec_typed, Cell, _}) -> lawspec_beam_handler:in_place(Cell, fun(S) -> {S, S} end);
state(_) -> erlang:error({lawspec, stateless_handler}).
set_state({lawspec_typed, Cell, _}, Value) ->
    lawspec_beam_handler:in_place(Cell, fun(_) -> {ok, Value} end);
set_state(_, _) ->
    erlang:error({lawspec, stateless_handler}).
