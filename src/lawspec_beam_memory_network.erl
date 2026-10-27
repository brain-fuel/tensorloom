%% @doc Owned in-memory packet transport with seeded loss, duplication and
%% delay. Frames can overtake each other; reliability belongs to the node
%% and channel protocol above this layer. No application code runs here.
%% ref:DEC-distribution-canonical-wire ref:DEC-portable-seeded-generation
-module(lawspec_beam_memory_network).
-behaviour(gen_server).
-export([
    start/1,
    with_network/2,
    stop/1,
    transport/2,
    insecure_transport_for_tests/2,
    register/3,
    unregister/2,
    send/4,
    partition/2,
    heal/1,
    recorded/1,
    trace/1
]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

%% Loss and duplication are probabilities; delay is seconds, as on the
%% other targets. BEAM timers round a positive delay up to a millisecond.
start(Options) ->
    lists:foreach(
        fun(Key) ->
            Value = maps:get(Key, Options, 0),
            valid(
                is_number(Value) andalso Value >= 0 andalso Value =< 1, {invalid_probability, Key}
            )
        end,
        [loss, duplicate]
    ),
    Delay = maps:get(delay, Options, 0),
    valid(is_number(Delay) andalso Delay >= 0 andalso Delay =< 4294967, invalid_delay),
    valid(is_integer(maps:get(seed, Options, 0)), invalid_seed),
    valid(is_boolean(maps:get(record, Options, false)), invalid_record),
    gen_server:start(?MODULE, {self(), Options}, []).
with_network(Options, Body) ->
    {ok, Network} = start(Options),
    try
        Body(Network)
    after
        stop(Network)
    end.
stop(Network) ->
    try
        gen_server:stop(Network, normal, infinity)
    catch
        exit:noproc -> ok;
        exit:{noproc, _} -> ok
    end.
transport(Network, Name) ->
    #{module => ?MODULE, network => Network, address => address(Name), insecure_for_tests => false}.
insecure_transport_for_tests(Network, Name) ->
    (transport(Network, Name))#{insecure_for_tests := true}.
register(Network, Address, Pid) ->
    gen_server:call(Network, {register, address(Address), Pid}, infinity).
unregister(Network, Address) -> gen_server:call(Network, {unregister, address(Address)}, infinity).
send(Network, Source, To, Bytes) ->
    gen_server:call(Network, {send, address(Source), address(To), Bytes}, infinity).
partition(Network, Groups) ->
    gen_server:call(
        Network, {partition, [[address(N) || N <- Group] || Group <- Groups]}, infinity
    ).
heal(Network) -> gen_server:call(Network, heal, infinity).
recorded(Network) -> gen_server:call(Network, recorded, infinity).
trace(Network) -> gen_server:call(Network, trace, infinity).
address(<<"mem://", _/binary>> = Address) -> Address;
address(Name) when is_binary(Name), byte_size(Name) > 0 -> <<"mem://", Name/binary>>.
valid(true, _) -> ok;
valid(false, Reason) -> error({lawspec, {memory_network, Reason}}).

init({Owner, Options}) ->
    {ok, #{
        owner => monitor(process, Owner),
        owner_pid => Owner,
        options => Options,
        random => lawspec_beam_random:seed(maps:get(seed, Options, 0)),
        nodes => #{},
        monitors => #{},
        pending => #{},
        groups => none,
        trace => []
    }}.
handle_call({register, Address, Pid}, {Caller, _}, State) ->
    case
        is_pid(Pid) andalso node(Pid) =:= node() andalso
            (Caller =:= Pid orelse Caller =:= maps:get(owner_pid, State))
    of
        false ->
            {reply, {error, not_node_owner}, State};
        true ->
            Current = prune(Address, State),
            Nodes = maps:get(nodes, Current),
            Monitors = maps:get(monitors, Current),
            case maps:find(Address, Nodes) of
                {ok, _} ->
                    {reply, {error, already_registered}, Current};
                error ->
                    Monitor = monitor(process, Pid),
                    {reply, ok, Current#{
                        nodes := Nodes#{Address => {Pid, Monitor}},
                        monitors := Monitors#{Monitor => Address}
                    }}
            end
    end;
handle_call({unregister, Address}, {Caller, _}, State = #{nodes := Nodes}) ->
    case maps:find(Address, Nodes) of
        error ->
            {reply, ok, State};
        {ok, {Pid, _}} ->
            case Caller =:= Pid orelse Caller =:= maps:get(owner_pid, State) of
                true -> {reply, ok, remove(Address, State)};
                false -> {reply, {error, not_node_owner}, State}
            end
    end;
handle_call({send, Source, To, Bytes}, {Caller, _}, State = #{nodes := Nodes}) ->
    case maps:find(Source, Nodes) of
        {ok, {Caller, _}} when is_binary(Bytes) ->
            {Result, Updated} = transmit(Source, To, Bytes, State),
            {reply, Result, Updated};
        _ ->
            {reply, {error, not_node_owner}, State}
    end;
handle_call({partition, Groups}, _, State) ->
    {reply, ok, State#{groups := Groups}};
handle_call(heal, _, State) ->
    {reply, ok, State#{groups := none}};
handle_call(recorded, _, State = #{trace := Trace}) ->
    {reply, [maps:get(frame, E) || E <- lists:reverse(Trace)], State};
handle_call(trace, _, State = #{trace := Trace}) ->
    {reply, lists:reverse(Trace), State}.
handle_cast(_, State) -> {noreply, State}.
handle_info({'DOWN', Monitor, process, _, _}, State = #{owner := Monitor}) ->
    {stop, normal, State};
handle_info({'DOWN', Monitor, process, _, _}, State = #{monitors := Monitors}) ->
    case maps:find(Monitor, Monitors) of
        {ok, Address} -> {noreply, remove(Address, State)};
        error -> {noreply, State}
    end;
handle_info(
    {deliver, Ref, Source, To, Generation, Bytes}, State = #{pending := Pending, nodes := Nodes}
) ->
    case {maps:is_key(Ref, Pending), maps:find(To, Nodes)} of
        {true, {ok, {Pid, Generation}}} -> Pid ! {lawspec_network, self(), Source, Bytes};
        _ -> ok
    end,
    {noreply, State#{pending := maps:remove(Ref, Pending)}};
handle_info(_, State) ->
    {noreply, State}.
terminate(_, #{pending := Pending}) ->
    maps:foreach(fun(_, {Timer, _, _}) -> erlang:cancel_timer(Timer) end, Pending),
    ok.

prune(Address, State = #{nodes := Nodes}) ->
    case maps:find(Address, Nodes) of
        {ok, {Pid, _}} ->
            case is_process_alive(Pid) of
                true -> State;
                false -> remove(Address, State)
            end;
        error ->
            State
    end.
remove(Address, State = #{nodes := Nodes, monitors := Monitors, pending := Pending}) ->
    case maps:take(Address, Nodes) of
        error ->
            State;
        {{_, Monitor}, Rest} ->
            demonitor(Monitor, [flush]),
            Remaining = maps:filter(
                fun(_, {Timer, To, Generation}) ->
                    case To =:= Address andalso Generation =:= Monitor of
                        true ->
                            erlang:cancel_timer(Timer),
                            false;
                        false ->
                            true
                    end
                end,
                Pending
            ),
            State#{nodes := Rest, monitors := maps:remove(Monitor, Monitors), pending := Remaining}
    end.
transmit(Source, To, Bytes, State) ->
    Event = #{source => Source, to => To, frame => Bytes},
    case maps:find(To, maps:get(nodes, State)) of
        error ->
            {{error, {unreachable, To}}, remember(Event#{outcome => unreachable}, State)};
        {ok, {_, Generation}} ->
            Groups = maps:get(groups, State),
            Reachable =
                Groups =:= none orelse
                    lists:any(
                        fun(G) -> lists:member(Source, G) andalso lists:member(To, G) end, Groups
                    ),
            case Reachable of
                false ->
                    {ok, remember(Event#{outcome => partitioned}, State)};
                true ->
                    {Lost, S1} = chance(loss, State),
                    case Lost of
                        true ->
                            {ok, remember(Event#{outcome => lost}, S1)};
                        false ->
                            {Duplicate, S2} = chance(duplicate, S1),
                            Copies =
                                case Duplicate of
                                    true -> 2;
                                    false -> 1
                                end,
                            {Slots, Next} = lists:mapfoldl(
                                fun(_, R) -> lawspec_beam_random:below(1001, R) end,
                                maps:get(random, S2),
                                lists:seq(1, Copies)
                            ),
                            Delay = maps:get(delay, maps:get(options, S2), 0),
                            Pending = lists:foldl(
                                fun(Slot, Acc) ->
                                    Ref = make_ref(),
                                    Timer = erlang:send_after(
                                        ceil(Slot * Delay),
                                        self(),
                                        {deliver, Ref, Source, To, Generation, Bytes}
                                    ),
                                    Acc#{Ref => {Timer, To, Generation}}
                                end,
                                maps:get(pending, S2),
                                Slots
                            ),
                            {ok,
                                remember(Event#{outcome => sent, delay_slots => Slots}, S2#{
                                    random := Next, pending := Pending
                                })}
                    end
            end
    end.
chance(Key, State = #{options := Options, random := Random}) ->
    Probability = maps:get(Key, Options, 0),
    case Probability > 0 of
        false ->
            {false, State};
        true ->
            {Draw, Next} = lawspec_beam_random:below(1 bsl 30, Random),
            {Draw < Probability * (1 bsl 30), State#{random := Next}}
    end.
remember(Event, State = #{options := Options, trace := Trace}) ->
    case maps:get(record, Options, false) of
        true -> State#{trace := [Event | Trace]};
        false -> State
    end.
