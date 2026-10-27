%% @doc Named frame routing and reliable request/reply over a transport.
%% Requests are deduplicated by sender and identity, including while their
%% handler is running. Application handlers run in owned, cancellable tasks.
%% ref:DEC-distribution-canonical-wire ref:DEC-async-native-tasks
-module(lawspec_beam_node).
-behaviour(gen_server).
-export([
    start/1, start/2,
    start_owned/2,
    with_node/3,
    stop/1,
    address/1,
    identity/1,
    register_handler/3,
    register_receiver/3,
    register_service/2,
    adopt_service/2,
    unregister/2,
    send/4,
    forward/2,
    request/5,
    request_async/5,
    cancel/2,
    reply/5
]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

start(Transport) -> start(Transport, #{}).
start(Transport, Options) -> start_as(none, Transport, Options).
start_owned(Transport, Options) -> start_as(self(), Transport, Options).
with_node(Transport, Options, Body) ->
    {ok, Node} = start_owned(Transport, Options),
    try
        Body(Node)
    after
        stop(Node)
    end.
start_as(Owner, Transport, Options) ->
    %% Only the memory transport has an explicit cleartext testing mode.
    %% The optional security module is emitted by import lawspec.network.
    Security =
        case Transport of
            #{module := lawspec_beam_memory_network, insecure_for_tests := true} -> test;
            _ -> code:ensure_loaded(lawspec_beam_network)
        end,
    case Security of
        Missing when Missing =/= test, Missing =/= {module, lawspec_beam_network} ->
            {error, secure_network_not_available};
        _ ->
            %% A replacement client may reuse its address while a peer still
            %% caches old replies. Its request stream must not restart at 1.
            <<Random:64/unsigned>> = crypto:strong_rand_bytes(8),
            First = maps:get(first_id, Options, (Random band ((1 bsl 61) - 1)) + 1),
            case is_integer(First) andalso First > 0 andalso First =< 16#ffffffffffffffff of
                true ->
                    gen_server:start(?MODULE, {Owner, Transport, Options#{first_id => First}}, []);
                false ->
                    {error, invalid_request_identity}
            end
    end.
stop(Node) ->
    try
        gen_server:stop(Node, normal, infinity)
    catch
        exit:noproc -> ok;
        exit:{noproc, _} -> ok
    end.
address(Node) -> call(Node, address).
identity(Node) -> call(Node, identity).
register_handler(Node, Name, Handler) when is_function(Handler, 1) ->
    Context = lists:keydelete(
        {lawspec_beam_tasks, scopes}, 1, lawspec_beam_runtime:worker_context()
    ),
    call(Node, {register, Name, {handler, Handler, Context}}).
register_receiver(Node, Name, Receiver) when is_pid(Receiver) ->
    call(Node, {register, Name, {receiver, Receiver}}).
%% Internal endpoint services belong to the node's cancellation scope.
%% Ordinary receivers remain owned by their application.
register_service(Node, Name) -> call(Node, {register, Name, {service, self()}}).
%% Session coordinators and relay workers have no wire address of their own.
adopt_service(Node, Service) when is_pid(Service) -> call(Node, {adopt_service, Service}).
unregister(Node, Name) -> call(Node, {unregister, Name}).
send(Node, Address, Kind, Payload) ->
    call(Node, {send, #{to => Address, kind => Kind, payload => Payload}}).
forward(Node, Frame) -> call(Node, {send, Frame}).
request(Node, Address, Kind, Payload, Timeout) ->
    call(Node, {request, sync, Address, Kind, Payload, Timeout}).
%% The caller receives {lawspec_reply, Node, Ticket, {ok, {Status, Bytes}}}
%% or {error, Reason}. The wire identity lets scenario clocks follow exactly
%% their own mailbox request, even when distinct requests overtake each other.
request_async(Node, Address, Kind, Payload, Timeout) ->
    call(Node, {request, async, Address, Kind, Payload, Timeout}).
cancel(Node, Ticket) -> call(Node, {cancel, Ticket}).
reply(Node, Source, Identity, Status, Body) -> call(Node, {reply, Source, Identity, Status, Body}).
call(Node, Request) ->
    Result =
        try
            gen_server:call(Node, Request, infinity)
        catch
            exit:{_, {gen_server, call, [Node, _, infinity]}} -> {error, node_closed}
        end,
    case Result of
        {ok, Value} -> Value;
        {error, Reason} -> error({lawspec, {network, Reason}})
    end.

init({Owner, Transport, Options}) ->
    Open =
        case Transport of
            #{module := lawspec_beam_memory_network, insecure_for_tests := true} ->
                lawspec_beam_transport:open(Transport, self());
            _ ->
                case lawspec_beam_network:start(self(), Transport, Options) of
                    {ok, Layer} ->
                        {ok, #{
                            module => lawspec_beam_network,
                            pid => Layer,
                            address => lawspec_beam_network:address(Layer)
                        }};
                    Error ->
                        Error
                end
        end,
    case Open of
        {error, Reason} ->
            {stop, {transport_registration, Reason}};
        {ok, Connection = #{pid := Network, address := Address}} ->
            Scope = lawspec_beam_tasks:open(),
            {ok, #{
                connection => Connection,
                address => Address,
                owner =>
                    case Owner of
                        none -> none;
                        _ -> monitor(process, Owner)
                    end,
                network_monitor => monitor(process, Network),
                scope => Scope,
                scope_monitor => monitor(process, Scope),
                entities => #{},
                entity_monitors => #{},
                pending => #{},
                pending_monitors => #{},
                seen => #{},
                jobs => #{},
                next_id => maps:get(first_id, Options)
            }}
    end.
handle_call(Request, From, State) ->
    try
        local(Request, From, State)
    catch
        throw:{invalid, Reason} -> {reply, {error, Reason}, State}
    end.
handle_cast(_, State) -> {noreply, State}.
handle_info(
    {lawspec_network, Network, _, Bytes},
    State = #{connection := #{module := lawspec_beam_memory_network, pid := Network}}
) ->
    receive_frame(Bytes, State);
handle_info(
    {lawspec_secure, Layer, _, Bytes},
    State = #{connection := #{module := lawspec_beam_network, pid := Layer}}
) ->
    receive_frame(Bytes, State);
handle_info({retry, Identity, Ticket}, State = #{pending := Pending}) ->
    case maps:find(Identity, Pending) of
        {ok, P = #{ticket := Ticket, deadline := Deadline}} ->
            Now = erlang:monotonic_time(millisecond),
            case Now >= Deadline of
                true ->
                    {noreply,
                        finish(Identity, {error, {unreachable, maps:get(address, P)}}, State)};
                false ->
                    _ = transmit(maps:get(frame, P), State),
                    Timer = erlang:send_after(
                        min(100, Deadline - Now), self(), {retry, Identity, Ticket}
                    ),
                    {noreply, State#{pending := Pending#{Identity := P#{timer := Timer}}}}
            end;
        _ ->
            {noreply, State}
    end;
handle_info({'DOWN', Monitor, process, _, _}, State = #{owner := Monitor}) ->
    {stop, normal, State};
handle_info({'DOWN', Monitor, process, _, _}, State = #{network_monitor := Monitor}) ->
    {stop, normal, State};
handle_info({'DOWN', Monitor, process, _, _}, State = #{scope_monitor := Monitor}) ->
    {stop, normal, State};
handle_info({'DOWN', Monitor, process, _, Reason}, State = #{jobs := Jobs}) ->
    case maps:take(Monitor, Jobs) of
        {Job, Rest} ->
            Outcome =
                case Reason of
                    {lawspec_network_result, Status, Body} -> {Status, Body};
                    _ -> {1, printable({handler_exited, Reason})}
                end,
            {noreply, answer_job(Job, Outcome, State#{jobs := Rest})};
        error ->
            {noreply, monitor_down(Monitor, State)}
    end;
handle_info(_, State) ->
    {noreply, State}.
terminate(_, State) ->
    maps:foreach(
        fun(_, P) ->
            erlang:cancel_timer(maps:get(timer, P)),
            respond(P, {error, node_closed})
        end,
        maps:get(pending, State)
    ),
    %% The scope also performs this cleanup if this server is killed before
    %% terminate/2 can run. close/1 joins every registered descendant.
    lawspec_beam_tasks:close(maps:get(scope, State)),
    lawspec_beam_transport:close(maps:get(connection, State)),
    ok.

receive_frame(Bytes, State) ->
    case lawspec_beam_wire:read_frame(Bytes) of
        {ok, Frame} -> {noreply, arrive(Frame, State)};
        {error, _} -> {noreply, State}
    end.

local(address, _, State) ->
    {reply, {ok, maps:get(address, State)}, State};
local(identity, _, State = #{connection := #{module := lawspec_beam_network, pid := Layer}}) ->
    {reply, {ok, lawspec_beam_network:identity(Layer)}, State};
local(identity, _, State) ->
    {reply, {error, insecure_node}, State};
local({adopt_service, Service}, _, State) ->
    ok = lawspec_beam_tasks:adopt(maps:get(scope, State), Service),
    {reply, {ok, ok}, State};
local({register, Name, Entity}, _, Previous) ->
    require(
        is_binary(Name) andalso byte_size(Name) > 0 andalso binary:match(Name, <<"/">>) =:= nomatch,
        invalid_entity_name
    ),
    %% A replacement may arrive before the receiver's DOWN notification.
    State = prune_entity(Name, Previous),
    #{entities := Entities, entity_monitors := Monitors} = State,
    require(not maps:is_key(Name, Entities), {already_registered, Name}),
    {Stored, UpdatedMonitors} =
        case Entity of
            {service, Pid} ->
                ok = lawspec_beam_tasks:adopt(maps:get(scope, State), Pid),
                M = monitor(process, Pid),
                {{receiver, Pid, M}, Monitors#{M => Name}};
            {receiver, Pid} ->
                M = monitor(process, Pid),
                {{receiver, Pid, M}, Monitors#{M => Name}};
            _ ->
                {Entity, Monitors}
        end,
    {reply, {ok, <<(maps:get(address, State))/binary, "/", Name/binary>>}, State#{
        entities := Entities#{Name => Stored}, entity_monitors := UpdatedMonitors
    }};
local({unregister, Name}, _, State) ->
    {reply, {ok, ok}, remove_entity(Name, State)};
local({send, Frame}, _, State) ->
    validate_frame(Frame),
    {reply, {ok, transmit(Frame, State)}, State};
local(
    {request, Mode, Address, Kind, Payload, Timeout},
    From = {Caller, _},
    State = #{next_id := Identity, pending := Pending, pending_monitors := Monitors}
) ->
    require(is_integer(Timeout) andalso Timeout >= 0, invalid_timeout),
    require(Identity =< 16#ffffffffffffffff, request_identity_exhausted),
    Frame = #{to => Address, kind => Kind, payload => Payload, id => Identity},
    validate_frame(Frame),
    {Peer, _} = lawspec_beam_wire:split_address(Address),
    Ticket = make_ref(),
    Monitor = monitor(process, Caller),
    Timer = erlang:send_after(min(100, Timeout), self(), {retry, Identity, Ticket}),
    Delivery =
        case Mode of
            sync -> {sync, From};
            async -> {async, Caller, self(), Ticket}
        end,
    P = #{
        address => Address,
        peer => Peer,
        frame => Frame,
        ticket => Ticket,
        caller => Caller,
        monitor => Monitor,
        timer => Timer,
        deadline => erlang:monotonic_time(millisecond) + Timeout,
        delivery => Delivery
    },
    _ = transmit(Frame, State),
    Next = State#{
        pending := Pending#{Identity => P},
        pending_monitors := Monitors#{Monitor => Identity},
        next_id := Identity + 1
    },
    case Mode of
        sync -> {noreply, Next};
        async -> {reply, {ok, {Ticket, Identity}}, Next}
    end;
local({cancel, Ticket}, {Caller, _}, State = #{pending := Pending}) ->
    Matches = [
        I
     || {I, #{ticket := T, caller := C}} <- maps:to_list(Pending), T =:= Ticket, C =:= Caller
    ],
    {reply, {ok, ok}, lists:foldl(fun(I, S) -> finish(I, cancelled, S) end, State, Matches)};
local({reply, Source, Identity, Status, Body}, {Caller, _}, State = #{seen := Seen}) ->
    validate_reply(Status, Body),
    Key = {Source, Identity},
    case maps:find(Key, Seen) of
        {ok, #{owner := Caller, response := pending}} ->
            {reply, {ok, ok}, answer(Key, Status, Body, State)};
        {ok, #{owner := Caller}} ->
            {reply, {ok, ok}, State};
        _ when Identity =:= 0 -> {reply, {ok, ok}, State};
        _ ->
            {reply, {error, not_request_owner}, State}
    end.
require(true, _) -> ok;
require(false, Reason) -> throw({invalid, Reason}).
validate_reply(Status, Body) ->
    require(
        is_integer(Status) andalso Status >= 0 andalso Status =< 255 andalso is_binary(Body),
        invalid_reply
    ).
validate_frame(Frame) ->
    try
        {_, _} = lawspec_beam_wire:split_address(maps:get(to, Frame)),
        _ = lawspec_beam_wire:frame(
            maps:get(kind, Frame),
            <<>>,
            maps:get(source, Frame, <<>>),
            maps:get(id, Frame, 0),
            maps:get(payload, Frame)
        ),
        ok
    catch
        _:_ -> throw({invalid, invalid_frame})
    end.
transmit(Frame, State = #{connection := Connection}) ->
    {Peer, Name} = lawspec_beam_wire:split_address(maps:get(to, Frame)),
    Bytes = lawspec_beam_wire:frame(
        maps:get(kind, Frame),
        Name,
        maps:get(source, Frame, maps:get(address, State)),
        maps:get(id, Frame, 0),
        maps:get(payload, Frame)
    ),
    lawspec_beam_transport:send(Connection, Peer, Bytes).

arrive(
    #{kind := <<"reply">>, id := Identity, source := Source, payload := <<Status, Body/binary>>},
    State = #{pending := Pending}
) ->
    case maps:find(Identity, Pending) of
        {ok, #{peer := Source}} -> finish(Identity, {ok, {Status, Body}}, State);
        _ -> State
    end;
arrive(#{kind := <<"reply">>}, State) ->
    State;
arrive(Frame = #{to := Name, source := Source, id := Identity}, State = #{seen := Seen}) ->
    Key = {Source, Identity},
    case maps:find(Key, Seen) of
        {ok, #{response := Response, fingerprint := Fingerprint}} when Identity =/= 0 ->
            case fingerprint(Frame) =:= Fingerprint of
                false ->
                    send_reply(
                        Source,
                        Identity,
                        <<3, "a request identity was reused for different content">>,
                        State
                    );
                true ->
                    case Response of
                        pending -> ok;
                        _ -> send_reply(Source, Identity, Response, State)
                    end
            end,
            State;
        _ ->
            dispatch(Name, Key, Frame, State)
    end.
dispatch(Name, Key = {Source, Identity}, Frame, State = #{entities := Entities, seen := Seen}) ->
    case maps:find(Name, Entities) of
        error ->
            case Identity of
                0 ->
                    State;
                _ ->
                    Entry = #{
                        owner => none,
                        entity => Name,
                        response => pending,
                        fingerprint => fingerprint(Frame)
                    },
                    answer(Key, 3, <<"nothing is registered as ", Name/binary>>, State#{
                        seen := Seen#{Key => Entry}
                    })
            end;
        {ok, {receiver, Pid, _}} ->
            Pid ! {lawspec_frame, self(), Frame},
            case Identity of
                0 ->
                    State;
                _ ->
                    State#{
                        seen := Seen#{
                            Key => #{
                                owner => Pid,
                                entity => Name,
                                response => pending,
                                fingerprint => fingerprint(Frame)
                            }
                        }
                    }
            end;
        {ok, {handler, Handler, Context}} ->
            Scope = maps:get(scope, State),
            WorkerContext = [{{lawspec_beam_tasks, scopes}, [Scope]} | Context],
            {Pid, Monitor} = spawn_monitor(fun() ->
                {Status, Body} =
                    try
                        lawspec_beam_runtime:with_worker_context(WorkerContext, fun() ->
                            lawspec_beam_tasks:with_scope(fun(_) ->
                                {Code, Bytes} = Handler(Frame),
                                validate_reply(Code, Bytes),
                                {Code, Bytes}
                            end)
                        end)
                    catch
                        Class:Reason -> {1, printable({Class, Reason})}
                    end,
                exit({lawspec_network_result, Status, Body})
            end),
            Jobs = maps:get(jobs, State),
            S1 = State#{jobs := Jobs#{Monitor => #{pid => Pid, key => Key}}},
            case Identity of
                0 ->
                    S1;
                _ ->
                    S1#{
                        seen := Seen#{
                            Key => #{
                                owner => Pid,
                                entity => Name,
                                response => pending,
                                source => Source,
                                fingerprint => fingerprint(Frame)
                            }
                        }
                    }
            end
    end.
answer_job(#{key := {_, 0}}, _, State) ->
    State;
answer_job(#{key := Key, pid := Pid}, {Status, Body}, State = #{seen := Seen}) ->
    case maps:find(Key, Seen) of
        {ok, #{owner := Pid, response := pending}} -> answer(Key, Status, Body, State);
        _ -> State
    end.
answer(Key = {Source, Identity}, Status, Body, State = #{seen := Seen}) ->
    Entry = maps:get(Key, Seen),
    Payload = <<Status, Body/binary>>,
    send_reply(Source, Identity, Payload, State),
    State#{seen := Seen#{Key := Entry#{response := Payload}}}.
send_reply(Source, Identity, Payload, State) ->
    %% A frame source is untrusted input even on the test-only transport.
    Frame = #{
        to => <<Source/binary, "/">>, kind => <<"reply">>, id => Identity, payload => Payload
    },
    try
        validate_frame(Frame),
        transmit(Frame, State)
    catch
        _:_ -> ok
    end.
finish(Identity, Result, State = #{pending := Pending, pending_monitors := Monitors}) ->
    case maps:take(Identity, Pending) of
        error ->
            State;
        {P = #{monitor := Monitor, timer := Timer}, Rest} ->
            erlang:cancel_timer(Timer),
            demonitor(Monitor, [flush]),
            case Result of
                cancelled -> ok;
                _ -> respond(P, Result)
            end,
            State#{pending := Rest, pending_monitors := maps:remove(Monitor, Monitors)}
    end.
respond(#{delivery := {sync, From}}, Result) ->
    gen_server:reply(From, Result);
respond(#{delivery := {async, Caller, Node, Ticket}}, Result) ->
    Caller ! {lawspec_reply, Node, Ticket, Result}.
monitor_down(Monitor, State = #{pending_monitors := Pending, entity_monitors := Entities}) ->
    case maps:find(Monitor, Pending) of
        {ok, Identity} ->
            finish(Identity, cancelled, State);
        error ->
            case maps:find(Monitor, Entities) of
                {ok, Name} -> remove_entity(Name, State);
                error -> State
            end
    end.
remove_entity(Name, State = #{entities := Entities, entity_monitors := Monitors, seen := Seen}) ->
    case maps:take(Name, Entities) of
        error ->
            State;
        {Entity, Rest} ->
            Ms =
                case Entity of
                    {receiver, _, M} ->
                        demonitor(M, [flush]),
                        maps:remove(M, Monitors);
                    _ ->
                        Monitors
                end,
            S1 = State#{entities := Rest, entity_monitors := Ms},
            lists:foldl(
                fun(Key, S) -> answer(Key, 2, <<"the receiving entity has stopped">>, S) end,
                S1,
                [K || {K, #{entity := N, response := pending}} <- maps:to_list(Seen), N =:= Name]
            )
    end.
prune_entity(Name, State = #{entities := Entities}) ->
    case maps:find(Name, Entities) of
        {ok, {receiver, Pid, _}} when node(Pid) =:= node() ->
            case is_process_alive(Pid) of
                true -> State;
                false -> remove_entity(Name, State)
            end;
        _ ->
            State
    end.
printable(Term) -> unicode:characters_to_binary(io_lib:format("~tp", [Term])).
fingerprint(#{kind := Kind, to := To, source := Source, id := Identity, payload := Payload}) ->
    crypto:hash(sha256, lawspec_beam_wire:frame(Kind, To, Source, Identity, Payload)).
