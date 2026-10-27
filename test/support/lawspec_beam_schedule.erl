%% @doc Seeded law ordering and native EUnit concurrency. Count live case
%% processes, including abnormal exits, instead of reporting CPU capacity as
%% achieved parallelism. The same ordering is used by the ExUnit driver.
%% ref:REQ-harness-units ref:DEC-portable-seeded-generation
-module(lawspec_beam_schedule).
-behaviour(gen_server).
-export([order/3, eunit/4, report/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

order(_, false, Items) ->
    Items;
order(Unit, true, Items) ->
    Seed = seed(),
    io:format(standard_error, "order random seed ~B (~ts): LAWSPEC_SEED=~B replays this order~n", [
        Seed, Unit, Seed
    ]),
    [
        Item
     || {_, _, Item} <- lists:sort([
            {
                crypto:hash(sha256, [integer_to_binary(Seed), 0, Unit, 0, integer_to_binary(Index)]),
                Index,
                Item
            }
         || {Index, Item} <- lists:enumerate(Items)
        ])
    ].

seed() ->
    case os:getenv("LAWSPEC_SEED") of
        false -> chosen_seed();
        "" -> chosen_seed();
        Text -> list_to_integer(Text)
    end.
chosen_seed() ->
    Key = {?MODULE, seed},
    case persistent_term:get(Key, undefined) of
        undefined ->
            Seed = binary:decode_unsigned(crypto:strong_rand_bytes(8)),
            persistent_term:put(Key, Seed),
            Seed;
        Seed ->
            Seed
    end.

eunit(Unit, Random, false, Groups) ->
    {inorder, order(Unit, Random, Groups)};
eunit(Unit, Random, true, Groups) ->
    Ordered = order(Unit, Random, Groups),
    {setup,
        fun() ->
            {ok, Observer} = gen_server:start(?MODULE, self(), []),
            Observer
        end,
        fun(Observer) ->
            Peak = gen_server:call(Observer, close, infinity),
            report(Unit, Peak)
        end,
        fun(Observer) ->
            %% Groups are law blocks; every checked example, boundary and
            %% native property remains a separately named native EUnit test.
            Cases = [{Label, [Test]} || {Label, Tests} <- Ordered, Test <- Tests],
            {inparallel, max(2, erlang:system_info(schedulers_online)), [
                observe(Observer, Test)
             || Test <- Cases
            ]}
        end}.

observe(Observer, {timeout, Seconds, Test}) ->
    {timeout, Seconds, observe(Observer, Test)};
observe(Observer, {Label, [Test]}) ->
    {Label, [observe(Observer, Test)]};
observe(_, {generator, _} = Skip) ->
    Skip;
observe(Observer, {Label, Test}) when is_function(Test, 0) ->
    {Label, fun() ->
        ok = gen_server:call(Observer, {enter, self()}, infinity),
        try
            Test()
        after
            gen_server:call(Observer, {leave, self()}, infinity)
        end
    end}.

report(Unit, Peak) ->
    lawspec_beam_harness:record(#{parallel => Unit, mode => <<"BEAM processes">>, workers => Peak}),
    io:format(standard_error, "~ts runs in parallel: BEAM processes, ~B worker(s)~n", [Unit, Peak]).

init(Owner) -> {ok, #{owner => monitor(process, Owner), active => #{}, peak => 0}}.
handle_call({enter, Pid}, _, State = #{active := Active, peak := Peak}) ->
    %% Remove already dead processes even if their DOWN message is queued
    %% behind this call, so a killed test cannot inflate the peak count.
    Live = maps:filter(
        fun(Process, Monitor) ->
            case is_process_alive(Process) of
                true ->
                    true;
                false ->
                    demonitor(Monitor, [flush]),
                    false
            end
        end,
        Active
    ),
    Added = Live#{Pid => monitor(process, Pid)},
    {reply, ok, State#{active := Added, peak := max(Peak, map_size(Added))}};
handle_call({leave, Pid}, _, State = #{active := Active}) ->
    case maps:take(Pid, Active) of
        {Monitor, Rest} ->
            demonitor(Monitor, [flush]),
            {reply, ok, State#{active := Rest}};
        error ->
            {reply, ok, State}
    end;
handle_call(close, _, State) ->
    {stop, normal, maps:get(peak, State), State}.
handle_cast(_, State) -> {noreply, State}.
handle_info({'DOWN', Owner, process, _, _}, State = #{owner := Owner}) ->
    {stop, normal, State};
handle_info({'DOWN', _, process, Pid, _}, State = #{active := Active}) ->
    {noreply, State#{active := maps:remove(Pid, Active)}};
handle_info(_, State) ->
    {noreply, State}.
