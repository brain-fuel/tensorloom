%% @doc The shared distribution frame format. All identifiers stay binaries;
%% a received frame never creates a BEAM atom. Values use the same canonical
%% descriptor codec as generated definitions and the other targets.
%% ref:DEC-distribution-canonical-wire
-module(lawspec_beam_wire).
-export([
    frame/5,
    read_frame/1,
    fields/3,
    read_fields/3,
    channel/3,
    read_channel/1,
    sequence/1,
    read_sequence/1,
    split_address/1
]).

frame(Kind, To, Source, Identity, Payload) ->
    fields(frame_types(), [Kind, To, Source, Identity, Payload], #{}).
read_frame(Bytes) ->
    try
        {Fields, <<>>} = read_fields(frame_types(), Bytes, #{}),
        [Kind, To, Source, Identity, Payload] = Fields,
        {ok, #{kind => Kind, to => To, source => Source, id => Identity, payload => Payload}}
    catch
        _:_ -> {error, invalid_frame}
    end.
frame_types() ->
    [
        [<<"text">>],
        [<<"text">>],
        [<<"text">>],
        [<<"int">>, <<"UInt64">>, 0, 16#ffffffffffffffff],
        [<<"bytes">>]
    ].
sequence_type() -> [<<"int">>, <<"Int64">>, -16#8000000000000000, 16#7fffffffffffffff].
fields(Types, Values, Table) ->
    true = length(Types) =:= length(Values),
    iolist_to_binary([lawspec_beam_values:encode(D, V, Table) || {D, V} <- lists:zip(Types, Values)]).
read_fields(Types, Bytes, Table) ->
    lists:mapfoldl(
        fun(D, Rest) -> lawspec_beam_values:decode_prefix(D, Rest, Table) end, Bytes, Types
    ).
channel(Sequence, Sender, Body) when is_binary(Body) ->
    Prefix = fields([sequence_type(), [<<"text">>]], [Sequence, Sender], #{}),
    <<Prefix/binary, Body/binary>>.
read_channel(Bytes) ->
    {[Sequence, Sender], Body} = read_fields([sequence_type(), [<<"text">>]], Bytes, #{}),
    {Sequence, Sender, Body}.
sequence(Number) -> lawspec_beam_values:encode(sequence_type(), Number, #{}).
read_sequence(Bytes) -> lawspec_beam_values:decode(sequence_type(), Bytes, #{}).
split_address(Address) when is_binary(Address) ->
    case binary:matches(Address, <<"/">>) of
        [] ->
            error({lawspec, {invalid_address, Address}});
        Matches ->
            {At, 1} = lists:last(Matches),
            <<Node:At/binary, $/, Name/binary>> = Address,
            case binary:match(Node, <<"://">>) of
                nomatch -> error({lawspec, {invalid_address, Address}});
                _ -> {Node, Name}
            end
    end.
