%% @doc Actors and checked definitions share canonical request/reply values.
%% A selector is a message name or a definition's content hash. Node workers
%% own application execution and deduplicate retries before invoking it.
%% ref:DEC-distribution-canonical-wire ref:DEC-async-native-tasks
-module(lawspec_beam_remote).
-export([serve/4, connect/5, call/3]).
-export_type([remote/0]).

-opaque remote() :: {lawspec_remote, pid(), binary(), binary(), map(), non_neg_integer()}.

%% Table: binary selector => #{arguments => [descriptor text], result =>
%% descriptor text, invoke => fun([logical argument]) -> logical result}.
serve(Node, Name, Kind, Table) when Kind =:= <<"call">>; Kind =:= <<"eval">> ->
    Compiled = maps:map(
        fun(Selector, Entry) ->
            require(is_binary(Selector), invalid_selector),
            require(is_function(maps:get(invoke, Entry), 1), invalid_remote_function),
            compile(Entry)
        end,
        Table
    ),
    lawspec_beam_node:register_handler(Node, Name, fun(Frame) -> dispatch(Kind, Compiled, Frame) end).

%% Client signatures use the same arguments/result entries without invoke.
connect(Node, Address, Kind, Signatures, Timeout) when Kind =:= <<"call">>; Kind =:= <<"eval">> ->
    require(is_integer(Timeout) andalso Timeout >= 0, invalid_timeout),
    _ = lawspec_beam_wire:split_address(Address),
    Compiled = maps:map(
        fun(Selector, Entry) ->
            require(is_binary(Selector), invalid_selector),
            compile(Entry)
        end,
        Signatures
    ),
    {lawspec_remote, Node, Address, Kind, Compiled, Timeout}.
compile(#{arguments := Arguments, result := Result} = Entry) ->
    Entry#{
        arguments := [lawspec_beam_values:from_text(Text) || Text <- Arguments],
        result := lawspec_beam_values:from_text(Result)
    }.

call({lawspec_remote, Node, Address, Kind, Table, Timeout}, Selector, Arguments) ->
    Entry =
        case maps:find(Selector, Table) of
            {ok, E} -> E;
            error -> fail({unknown_selector, Selector})
        end,
    Types = maps:get(arguments, Entry),
    require(is_list(Arguments) andalso length(Arguments) =:= length(Types), argument_count),
    Payload = iolist_to_binary([
        lawspec_beam_values:encode([<<"text">>], Selector, #{}),
        [encode(Type, Value) || {Type, Value} <- lists:zip(Types, Arguments)]
    ]),
    case lawspec_beam_node:request(Node, Address, Kind, Payload, Timeout) of
        {0, Bytes} -> decode(maps:get(result, Entry), Bytes);
        {1, Message} -> fail({failed, Message});
        {2, Message} -> fail({stopped, Message});
        {3, Message} -> fail({rejected, Message});
        {Status, _} -> fail({invalid_reply_status, Status})
    end.

dispatch(Kind, Table, #{kind := Kind, payload := Bytes}) ->
    Parsed =
        try
            {Selector, Rest} = lawspec_beam_values:decode_prefix([<<"text">>], Bytes, #{}),
            Entry = maps:get(Selector, Table),
            {Arguments, <<>>} = lists:mapfoldl(
                fun({Types, Descriptor}, Remaining) ->
                    lawspec_beam_values:decode_prefix(Descriptor, Remaining, Types)
                end,
                Rest,
                maps:get(arguments, Entry)
            ),
            {ok, Entry, Arguments}
        catch
            _:_ -> invalid
        end,
    case Parsed of
        invalid ->
            {3, <<"unknown selector or malformed remote arguments">>};
        {ok, Found, Values} ->
            try
                Result = (maps:get(invoke, Found))(Values),
                {0, encode(maps:get(result, Found), Result)}
            catch
                error:{lawspec, actor_stopped} -> {2, <<"the actor has stopped">>};
                Class:Reason -> {1, printable({Class, Reason})}
            end
    end;
dispatch(_, _, _) ->
    {3, <<"this service does not handle that request kind">>}.

encode({Table, Descriptor}, Value) -> lawspec_beam_values:encode(Descriptor, Value, Table).
decode({Table, Descriptor}, Bytes) -> lawspec_beam_values:decode(Descriptor, Bytes, Table).
require(true, _) -> ok;
require(false, Reason) -> fail(Reason).
fail(Reason) -> error({lawspec, {remote, Reason}}).
printable(Term) -> unicode:characters_to_binary(io_lib:format("~tp", [Term])).
