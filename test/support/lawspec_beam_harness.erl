%% @doc Adequacy of generated property cases. Each native framework stops
%% generation at its first failure, then shrinks/rechecks that failure. Freeze
%% observations at that boundary; examples, deterministic cases, rejected
%% tuples and native shrink candidates never change the coverage denominator.
%% Statistics use the same LAWSPEC_STATS records as the other targets.
%% ref:REQ-harness-units ref:DEC-never-pass-vacuously
-module(lawspec_beam_harness).
-export([
    run/4, run/5,
    abort/1,
    mark_harness_failure/1,
    sample/2,
    record/1,
    higher/2,
    score_text/1,
    known_discrepancy/4,
    skip/2,
    erlang_cases/1,
    benchmark/3
]).

-define(STATE, {?MODULE, observations}).
-define(EXPECTED, {?MODULE, expected_failure}).

%% Real monotonic time; benchmarks never assert the value they measure and
%% never install a virtual workflow clock. Evaluate every iteration eagerly.
%% A callback exception stays a native failure and cannot publish a complete
%% timing record. Bound both the time budget and allocation for fast bodies.
benchmark(Unit, Name, Body) ->
    Started = erlang:monotonic_time(nanosecond),
    {Count, Total, Fastest} = benchmark_loop(Body, Started, 0, 0, none),
    Mean = Total div Count,
    record(#{
        benchmark => Name, unit => Unit, iterations => Count, mean_ns => Mean, min_ns => Fastest
    }),
    io:format(
        "benchmark ~ts: ~B iteration(s), mean ~.2f us, fastest ~.2f us~n",
        [Name, Count, Mean / 1000, Fastest / 1000]
    ),
    ok.

benchmark_loop(_, _, Count, Total, Fastest) when Count >= 100000 -> {Count, Total, Fastest};
benchmark_loop(Body, Started, Count, Total, Fastest) ->
    Before = erlang:monotonic_time(nanosecond),
    case Count >= 3 andalso Before - Started >= 200000000 of
        true ->
            {Count, Total, Fastest};
        false ->
            Body(),
            Elapsed = erlang:monotonic_time(nanosecond) - Before,
            Best =
                case Fastest of
                    none -> Elapsed;
                    _ -> min(Fastest, Elapsed)
                end,
            benchmark_loop(Body, Started, Count + 1, Total + Elapsed, Best)
    end.

%% EUnit has no public intentional-skip test descriptor. A generator can
%% report the skipped obligation and return an empty group, without running
%% a dummy passing test. ExUnit reports its native skip through a formatter.
erlang_cases(Cases) ->
    [
        case Case of
            {Label, {skip, Law, Reason}} ->
                {generator, fun() ->
                    skip(Law, Reason),
                    {Label, []}
                end};
            _ ->
                Case
        end
     || Case <- Cases
    ].

skip(Law, Reason) ->
    io:format(standard_error, "SKIPPED ~ts: ~ts~n", [Law, Reason]),
    commit(#{
        law => Law, test => <<Law/binary, " skipped">>, outcome => <<"skipped">>, reason => Reason
    }).

%% A law is expected to have a failing check, not a broken harness. Collect
%% its checked examples/boundaries/properties/searches into one report. A
%% satisfied expectation must not leave ordinary failed child reports behind.
known_discrepancy(Law, Name, Reason, Tests) ->
    Previous = put(?EXPECTED, #{law => Law, checks => [], aborted => none}),
    {Outcome, Reports, Aborted} =
        try
            Result =
                try
                    lists:foreach(fun expected_case/1, Tests),
                    passed
                catch
                    Kind:Cause:Stack -> {failed, Kind, Cause, Stack}
                end,
            #{checks := Checks, aborted := FailedHarness} = get(?EXPECTED),
            {Result, lists:reverse(Checks), FailedHarness}
        after
            case Previous of
                undefined -> erase(?EXPECTED);
                _ -> put(?EXPECTED, Previous)
            end
        end,
    Base = #{law => Law, test => Name, reason => Reason, checks => Reports},
    case {Outcome, Aborted} of
        {passed, none} ->
            commit(Base#{outcome => <<"known-discrepancy-passed">>}),
            io:format(
                standard_error,
                "~ts is marked known discrepancy (~ts), but it passes; review the specification and implementation before removing `known discrepancy` from its harness~n",
                [Law, Reason]
            ),
            error({lawspec, {known_discrepancy_passed, Law, Reason}});
        {{failed, Kind1, Cause1, _}, none} ->
            commit(Base#{
                outcome => <<"known-discrepancy">>, failure => exception_text(Kind1, Cause1)
            }),
            io:format("~ts has a known discrepancy (~ts): ~ts~n", [
                Law, Reason, exception_text(Kind1, Cause1)
            ]),
            ok;
        {{failed, Kind1, Cause1, Stack1}, _} ->
            try
                commit(Base#{
                    outcome => <<"failed">>,
                    failureKind => <<"harness">>,
                    failure => exception_text(Kind1, Cause1)
                })
            catch
                _:_ -> ok
            end,
            erlang:raise(Kind1, Cause1, Stack1);
        {passed, {failure, Cause1}} ->
            commit(Base#{outcome => <<"failed">>, failureKind => <<"harness">>}),
            error(Cause1)
    end.

expected_case({timeout, _, Test}) -> expected_case(Test);
expected_case({_, Test}) when is_function(Test, 0) -> Test().

exception_text(Kind, Cause) ->
    unicode:characters_to_binary(io_lib:format("~tp:~tp", [Kind, Cause])).

run(Law, Name, Covers, Test) ->
    run(Law, Name, Covers, #{}, Test).

%% Each repeat is a complete native check, with its own counters and shrink
%% state. A retry starts the whole repeat sequence again. Do not pool coverage
%% across runs: a later adequate run cannot repair an earlier shortfall.
run(Law, Name, Covers, Options, Test) ->
    Repeat = maps:get(repeat, Options, 1),
    Retries = maps:get(retries, Options, 0),
    true = is_integer(Repeat) andalso Repeat > 0,
    true = is_integer(Retries) andalso Retries >= 0,
    Previous = get(?STATE),
    try
        attempts(Law, Name, Covers, Options, Test, Repeat, Retries, 1, [])
    after
        case Previous of
            undefined -> erase(?STATE);
            _ -> put(?STATE, Previous)
        end
    end.

attempts(Law, Name, Covers, Options, Test, Repeat, Retries, Attempt, History) ->
    {Outcome, Retryable, Report, Runs} = repeats(
        Law, Name, Covers, Options, Test, Repeat, Attempt, 1, History
    ),
    case Outcome of
        {error, _, _, _} when Retryable, Attempt =< Retries ->
            attempts(Law, Name, Covers, Options, Test, Repeat, Retries, Attempt + 1, Runs);
        _ ->
            Status =
                case Outcome of
                    {ok, _} when Attempt > 1 -> <<"flaky">>;
                    {ok, _} -> <<"passed">>;
                    _ -> <<"failed">>
                end,
            Final0 = Report#{outcome => Status, attempts => Attempt},
            Final =
                case Retryable of
                    true -> Final0;
                    false -> Final0#{failureKind => <<"harness">>}
                end,
            Details =
                case Repeat =/= 1 orelse Retries =/= 0 of
                    true -> Final#{repeat => Repeat, runs => lists:reverse(Runs)};
                    false -> Final
                end,
            case Outcome of
                {ok, Result} ->
                    publish(Details),
                    case Status of
                        <<"flaky">> ->
                            io:format("~ts is flaky: it failed, then passed on attempt ~B~n", [
                                Law, Attempt
                            ]);
                        _ ->
                            ok
                    end,
                    Result;
                {error, Kind, Reason, Stack} ->
                    %% Reporting must not replace the native failure, its
                    %% smallest counterexample or its replay seed.
                    try
                        publish(Details)
                    catch
                        ReportKind:ReportReason ->
                            io:format(
                                standard_error,
                                "LawSpec statistics failed for ~ts: ~tp:~tp~n",
                                [Law, ReportKind, ReportReason]
                            )
                    end,
                    erlang:raise(Kind, Reason, Stack)
            end
    end.

repeats(Law, Name, Covers, Options, Test, Repeat, Attempt, Repetition, History) ->
    put(?STATE, #{
        law => Law,
        test => Name,
        phase => generating,
        cases => 0,
        cover => [0 || _ <- Covers],
        classes => #{},
        labels => #{},
        target => false,
        best => none,
        aborted => none,
        extra => #{}
    }),
    Result = execute(Law, Options, Test),
    State = get(?STATE),
    Observed = maps:get(observed, Options, true),
    Report = maps:merge(
        maps:get(extra, State),
        case Observed of
            true -> report(Law, Name, Covers, State);
            false -> #{law => Law, test => Name}
        end
    ),
    Unmet = [Row || #{met := false} = Row <- maps:get(cover, Report, [])],
    {Outcome, Retryable} =
        case {Result, maps:get(aborted, State), Unmet} of
            {{ok, _}, {failure, Cause}, _} ->
                {exception(Cause), false};
            {{ok, _}, _, [_ | _]} ->
                {exception({lawspec, {harness_failed, Law, {unmet_cover, Unmet}}}), false};
            {{error, _, _, _}, {failure, _}, _} ->
                {Result, false};
            _ ->
                {Result, true}
        end,
    Status =
        case Outcome of
            {ok, _} -> <<"passed">>;
            _ -> <<"failed">>
        end,
    Run = (maps:without([law, test, attempts], Report))#{
        attempt => Attempt, repetition => Repetition, outcome => Status
    },
    Runs = [Run | History],
    case Outcome of
        {ok, _} when Repetition < Repeat ->
            repeats(Law, Name, Covers, Options, Test, Repeat, Attempt, Repetition + 1, Runs);
        _ ->
            {Outcome, Retryable, Report, Runs}
    end.

exception(Reason) ->
    try
        error(Reason)
    catch
        error:Reason:Stack -> {error, error, Reason, Stack}
    end.

%% Run a whole native check in one worker, preserving its generator and shrink
%% state. The supervisor owns cancellation and the private-resource registry;
%% neither dies when the native runner or the test deadline kills the borrower.
execute(Law, Options, Test) ->
    Timeout = maps:get(timeout, Options, infinity),
    true = Timeout =:= infinity orelse is_integer(Timeout) andalso Timeout > 0,
    case Timeout =/= infinity orelse maps:get(resources, Options, false) of
        false ->
            capture(Test);
        true ->
            Caller = self(),
            Context = lawspec_beam_runtime:worker_context(),
            Snapshot = worker_state(),
            CleanupTimeout = maps:get(cleanup_timeout, Options, 5000),
            Suite =
                try
                    lawspec_beam_resources:current()
                catch
                    error:{lawspec, missing_resource_run} -> none
                end,
            {Supervisor, Monitor} = spawn_monitor(fun() ->
                supervise(Caller, Suite, Law, Timeout, CleanupTimeout, Context, Snapshot, Test)
            end),
            receive
                {'DOWN', Monitor, process, Supervisor, {test_result, Result, State, Failure}} ->
                    restore_worker_state(State),
                    case Failure of
                        none -> ok;
                        _ -> mark_harness_failure(Failure)
                    end,
                    Result;
                {'DOWN', Monitor, process, Supervisor, Reason} ->
                    Failure = {lawspec, {test_supervisor_failed, Law, Reason}},
                    mark_harness_failure(Failure),
                    exception(Failure)
            end
    end.

supervise(Caller, Suite, Law, Timeout, CleanupTimeout, Context, Snapshot, Test) ->
    Parent = monitor(process, Caller),
    ok = lawspec_beam_resources:watch_case(Suite),
    {ok, Resources} = lawspec_beam_resources:start(CleanupTimeout),
    Scope = lawspec_beam_tasks:open(),
    Supervisor = self(),
    Parents =
        case proplists:get_value({lawspec_beam_tasks, scopes}, Context) of
            undefined -> [];
            Ps -> Ps
        end,
    WorkerContext = lists:keystore(
        {lawspec_beam_tasks, scopes},
        1,
        Context,
        {{lawspec_beam_tasks, scopes}, [Scope | Parents]}
    ),
    {Worker, Monitor} = spawn_monitor(fun() ->
        restore_worker_state(Snapshot),
        put({?MODULE, supervisor}, Supervisor),
        Result = capture(fun() ->
            lawspec_beam_runtime:with_worker_context(WorkerContext, fun() ->
                lawspec_beam_resources:with_case_run(Resources, Test)
            end)
        end),
        exit({test_result, Result, worker_state()})
    end),
    Deadline =
        case Timeout of
            infinity -> infinity;
            _ -> erlang:monotonic_time(millisecond) + Timeout
        end,
    {Outcome, State, Failure} = await_test(
        Parent, Worker, Monitor, Deadline, Law, Timeout, Snapshot
    ),
    %% Join nested LawSpec tasks before releasing their borrowed resources.
    lawspec_beam_tasks:close(Scope),
    Cleanup = capture(fun() -> lawspec_beam_resources:close(Resources) end),
    ok = lawspec_beam_resources:finish_case(Suite, Cleanup),
    {Result, FinalFailure} =
        case Cleanup of
            {ok, ok} ->
                {Outcome, Failure};
            {error, Kind, Reason, Stack} ->
                Cause =
                    {lawspec,
                        {test_cleanup_failed, Law, {test, outcome_reason(Outcome)}, {Kind, Reason}}},
                {{error, error, Cause, Stack}, Cause}
        end,
    case is_process_alive(Caller) of
        true ->
            demonitor(Parent, [flush]),
            exit({test_result, Result, State, FinalFailure});
        false ->
            case Cleanup of
                {ok, ok} ->
                    ok;
                _ ->
                    io:format(
                        standard_error,
                        "LawSpec cleanup failed after test cancellation: ~tp~n",
                        [FinalFailure]
                    )
            end
    end.

await_test(Parent, Worker, Monitor, Deadline, Law, Timeout, Snapshot) ->
    Remaining =
        case Deadline of
            infinity -> infinity;
            _ -> max(0, Deadline - erlang:monotonic_time(millisecond))
        end,
    receive
        {test_state, Worker, State} ->
            await_test(Parent, Worker, Monitor, Deadline, Law, Timeout, State);
        {'DOWN', Monitor, process, Worker, {test_result, Result, State}} ->
            {Result, State, none};
        {'DOWN', Monitor, process, Worker, Reason} ->
            Failure = {lawspec, {test_worker_failed, Law, Reason}},
            {exception(Failure), Snapshot, Failure};
        {'DOWN', Parent, process, _, _} ->
            kill_worker(Worker, Monitor),
            Failure = {lawspec, {test_caller_stopped, Law}},
            {exception(Failure), latest_state(Worker, Snapshot), Failure};
        close_case ->
            kill_worker(Worker, Monitor),
            Failure = {lawspec, {test_run_closed, Law}},
            {exception(Failure), latest_state(Worker, Snapshot), Failure}
    after Remaining ->
        kill_worker(Worker, Monitor),
        Failure = {lawspec, {test_timeout, Law, Timeout}},
        {exception(Failure), latest_state(Worker, Snapshot), Failure}
    end.

kill_worker(Worker, Monitor) ->
    exit(Worker, kill),
    receive
        {'DOWN', Monitor, process, Worker, _} -> ok
    end.
latest_state(Worker, Snapshot) ->
    receive
        {test_state, Worker, State} -> latest_state(Worker, State)
    after 0 -> Snapshot
    end.
capture(Body) ->
    try
        {ok, Body()}
    catch
        Kind:Reason:Stack -> {error, Kind, Reason, Stack}
    end.
outcome_reason({ok, _}) -> passed;
outcome_reason({error, Kind, Reason, _}) -> {Kind, Reason}.
worker_state() -> [{Key, get(Key)} || Key <- [?STATE, ?EXPECTED]].
restore_worker_state(State) ->
    lists:foreach(
        fun
            ({Key, undefined}) -> erase(Key);
            ({Key, Value}) -> put(Key, Value)
        end,
        State
    ).
set_state(State) ->
    put(?STATE, State),
    case get({?MODULE, supervisor}) of
        undefined ->
            ok;
        Supervisor ->
            Supervisor ! {test_state, self(), worker_state()},
            ok
    end.

%% Native frameworks can catch and simplify a generator error while shrinking.
%% Mark a harness failure where it occurs, before its exception is wrapped or
%% reduced to false. The original native failure still reaches the caller.
abort(Reason) ->
    Failure = {lawspec, Reason},
    mark_harness_failure(Failure),
    error(Failure).

mark_harness_failure(Failure) ->
    case get(?STATE) of
        #{aborted := none} = State -> set_state(State#{aborted := {failure, Failure}});
        _ -> ok
    end,
    case get(?EXPECTED) of
        #{aborted := none} = Expected -> put(?EXPECTED, Expected#{aborted := {failure, Failure}});
        _ -> ok
    end,
    ok.

%% Native shrinking still evaluates the pure metadata so an exception in it
%% stays reproducible. Only accounting freezes. The law stays on the native
%% case process, preserving the framework's shrink state. Resource callbacks
%% have separate owners; cases borrow handles from those owners.
sample(Observe, Test) ->
    case get(?STATE) of
        #{phase := failed} ->
            Observe(),
            Test();
        #{cases := Count} = State ->
            set_state(State#{cases := Count + 1}),
            try
                observe(Observe()),
                case Test() of
                    true ->
                        true;
                    Other ->
                        failed(),
                        Other
                end
            catch
                Kind:Reason:Stack ->
                    failed(),
                    erlang:raise(Kind, Reason, Stack)
            end;
        undefined ->
            error({lawspec, harness_observation_outside_run})
    end.

failed() ->
    set_state((get(?STATE))#{phase := failed}),
    ok.

observe({Covers, Classes, Labels}) ->
    observe({Covers, Classes, Labels, none});
observe({Covers, Classes, Labels, Score}) ->
    #{cover := Counts, classes := OldClasses, labels := OldLabels} = State = get(?STATE),
    %% Cover clauses have independent counters even if their labels coincide.
    %% Classifications and Text labels describe shares of cases, so the same
    %% label occurring twice in one case still counts that case only once.
    Hits = [Label || {Label, Holds} <- Classes, boolean(Holds)],
    AllClasses = maps:merge(maps:from_list([{Label, 0} || {Label, _} <- Classes]), OldClasses),
    set_state(State#{
        cover := cover_hits(Covers, Counts),
        classes := count(lists:usort(Hits), AllClasses),
        labels := count(lists:usort(Labels), OldLabels)
    }),
    observe_score(Score).

observe_score(none) ->
    ok;
observe_score({score, Score}) ->
    #{best := Best} = State = get(?STATE),
    set_state(State#{
        target := true,
        best :=
            case higher(Score, Best) of
                true -> {score, Score};
                false -> Best
            end
    }),
    ok.

%% Scores retain exact integers, decimals and rationals; an IEEE NaN has no
%% order and cannot be a best score. Converting through a native float would
%% collapse adjacent large integers and could make the climb stop too early.
%% ref:DEC-portable-exact-arithmetic
higher({ls_float, _, _} = Score, Best) ->
    case lawspec_beam_scalar:float_class(Score) of
        nan ->
            false;
        _ ->
            case Best of
                none -> true;
                {score, Other} -> lawspec_beam_scalar:float_compare(Score, Other) =:= 1
            end
    end;
higher(Score, Best) when
    is_integer(Score); element(1, Score) =:= ls_ratio; element(1, Score) =:= ls_decimal
->
    case Best of
        none -> true;
        {score, Other} -> lawspec_beam_scalar:compare(Score, Other) =:= 1
    end;
higher(Score, _) ->
    error({lawspec, {invalid_target_score, Score}}).

score_text(N) when is_integer(N) -> integer_to_binary(N);
score_text({ls_ratio, N, D}) ->
    iolist_to_binary([integer_to_binary(N), "/", integer_to_binary(D)]);
score_text({ls_decimal, C, E}) ->
    iolist_to_binary([integer_to_binary(C), "e", integer_to_binary(E)]);
score_text({ls_float, W, B} = Score) ->
    case lawspec_beam_scalar:float_class(Score) of
        nan ->
            <<"NaN">>;
        infinity ->
            case B bsr (W - 1) of
                0 -> <<"Infinity">>;
                _ -> <<"-Infinity">>
            end;
        _ ->
            float_to_binary(lawspec_beam_scalar:float_to_native(Score), [short])
    end.

cover_hits([], []) ->
    [];
cover_hits([Holds | Rest], [N | Counts]) ->
    [
        N +
            case boolean(Holds) of
                true -> 1;
                false -> 0
            end
        | cover_hits(Rest, Counts)
    ];
cover_hits(_, _) ->
    error({lawspec, invalid_harness_cover_observation}).

boolean(true) -> true;
boolean(false) -> false;
boolean(Value) -> error({lawspec, {invalid_harness_predicate, Value}}).

count(Labels, Counts) ->
    lists:foldl(
        fun(Label, Acc) when is_binary(Label) ->
            maps:update_with(Label, fun(N) -> N + 1 end, 1, Acc)
        end,
        Counts,
        Labels
    ).

report(
    Law,
    Name,
    Covers,
    #{cases := Cases, cover := Hits, classes := Classes, labels := Labels} = State
) ->
    Rows = [
        #{
            label => Label,
            required => Percent,
            observed => percentage(N, Cases),
            hits => N,
            cases => Cases,
            %% Use exact counts, not rounded percentages, to decide adequacy.
            met => Cases > 0 andalso 100 * N >= Percent * Cases
        }
     || {{Percent, Label}, N} <- lists:zip(Covers, Hits)
    ],
    Report = #{
        law => Law,
        test => Name,
        attempts => 1,
        cases => Cases,
        cover => Rows,
        classes => Classes,
        labels => Labels
    },
    case maps:get(target, State) of
        false ->
            Report;
        true ->
            Report#{
                target => #{
                    best =>
                        case maps:get(best, State) of
                            none -> null;
                            {score, Score} -> score_text(Score)
                        end
                }
            }
    end.

percentage(_, 0) -> 0.0;
percentage(N, Cases) -> round(10000 * N / Cases) / 100.

publish(
    #{law := Law, cases := Cases, cover := Covers, classes := Classes, labels := Labels} = Report
) ->
    io:format("~ts: ~B generated case(s)~n", [Law, Cases]),
    lists:foreach(
        fun(
            #{label := Label, required := Required, observed := Observed, met := Met, hits := Hits}
        ) ->
            io:format("  cover ~B% \"~ts\": ~.2f%~ts~n", [
                Required,
                Label,
                Observed,
                case Met of
                    true -> "";
                    false -> io_lib:format(" (not met; ~B/~B cases)", [Hits, Cases])
                end
            ])
        end,
        Covers
    ),
    lists:foreach(
        fun({Label, N}) -> io:format("  ~ts: ~.2f%~n", [Label, percentage(N, Cases)]) end,
        lists:sort(maps:to_list(Classes))
    ),
    lists:foreach(
        fun({Label, N}) -> io:format("  label ~ts: ~.2f%~n", [Label, percentage(N, Cases)]) end,
        lists:sort(maps:to_list(Labels))
    ),
    case Report of
        #{target := #{best := Best}} when Best =/= null ->
            io:format("  best target score: ~ts~n", [Best]);
        _ ->
            ok
    end,
    commit(Report);
publish(Report) ->
    commit(Report).

record(#{law := Law, test := Name} = Report) ->
    %% A search runs inside the same repeat/retry boundary as other tests.
    %% Fold its report into that run instead of overwriting the final status
    %% or losing a prior failed attempt when the next attempt succeeds.
    case get(?STATE) of
        #{law := Law, test := Name, extra := Extra} = State ->
            set_state(State#{extra := maps:merge(Extra, Report)}),
            ok;
        _ ->
            commit(Report)
    end;
record(#{parallel := Unit} = Report) ->
    write_record({parallel, Unit}, Report);
record(#{benchmark := Name, unit := Unit} = Report) ->
    write_record({benchmark, Unit, Name}, Report).

commit(#{law := Law} = Report) ->
    case get(?EXPECTED) of
        #{law := Law, checks := Checks} = Expected ->
            Aborted =
                case Report of
                    #{failureKind := <<"harness">>} -> {failure, {lawspec, {harness_failed, Law}}};
                    _ -> maps:get(aborted, Expected)
                end,
            put(?EXPECTED, Expected#{checks := [Report | Checks], aborted := Aborted}),
            ok;
        _ ->
            write_record(Report)
    end.

write_record(#{law := Law, test := Name} = Report) ->
    write_record({Law, Name}, Report).

write_record(Identity, Report) ->
    case os:getenv("LAWSPEC_STATS") of
        false ->
            ok;
        "" ->
            ok;
        Directory ->
            %% Stable full-identity hash avoids both path traversal and the
            %% collisions caused by replacing punctuation with underscores.
            Hash = binary_to_list(binary:encode_hex(crypto:hash(sha256, term_to_binary(Identity)))),
            Path = filename:join(Directory, Hash ++ ".json"),
            Temporary =
                Path ++ "." ++ os:getpid() ++ "." ++
                    integer_to_list(erlang:unique_integer([positive])) ++ ".tmp",
            ok = filelib:ensure_dir(Path),
            try
                ok = file:write_file(Temporary, json:encode(Report), [exclusive]),
                ok = file:rename(Temporary, Path)
            after
                file:delete(Temporary)
            end
    end.
