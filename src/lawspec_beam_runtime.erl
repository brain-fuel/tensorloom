%% @doc Shared execution services for generated BEAM definitions and laws.
%% Values stay in the portable domain until an explicit native boundary.
%% ref:DEC-typed-core-boundary
-module(lawspec_beam_runtime).
-export([
    bind/2,
    require/2,
    assert_equal/3,
    contextual/2,
    helper/4,
    concurrently/1,
    async_call/1,
    worker_context/0,
    with_worker_context/2,
    recorded/4,
    search_elements/3
]).

%% A three-valued finite search, in candidate order and with short circuiting.
search_elements(Universal, Values, Evaluate) ->
    Decisive =
        case Universal of
            true -> -1;
            false -> 1
        end,
    search_elements(Values, Evaluate, Decisive, -Decisive).

search_elements([], _Evaluate, _Decisive, Result) ->
    Result;
search_elements([Value | Rest], Evaluate, Decisive, Result) ->
    case Evaluate(Value) of
        Decisive -> Decisive;
        0 -> search_elements(Rest, Evaluate, Decisive, 0);
        Code when Code =:= -Decisive -> search_elements(Rest, Evaluate, Decisive, Result);
        _ -> erlang:error(invalid_finite_search_outcome)
    end.

%% LawSpec's allocator, active handler operation, cancellation scopes and
%% workflow runtime/frame cross workers. Application entries stay process-local.
worker_context() ->
    [
        {Key, get(Key)}
     || Key <-
            [
                {lawspec_beam_effects, scope},
                {lawspec_beam_handler, context},
                {lawspec_beam_tasks, scopes},
                {lawspec_beam_workflow, runtime},
                {lawspec_beam_workflow, frame},
                {lawspec_beam_resources, run},
                {lawspec_beam_resources, case_run}
            ]
    ].

with_worker_context(Context, Body) ->
    Previous = [{Key, put(Key, Value)} || {Key, Value} <- Context],
    try
        case get({lawspec_beam_tasks, scopes}) of
            undefined -> ok;
            Scopes -> lawspec_beam_tasks:attach(Scopes)
        end,
        Body()
    after
        lists:foreach(
            fun
                ({Key, undefined}) -> erase(Key);
                ({Key, Value}) -> put(Key, Value)
            end,
            Previous
        )
    end.

%% let x = value in body, as generated code writes it: the value, then the
%% body given it.
bind(Value, Body) -> Body(Value).

require(true, _) -> ok;
require(false, Context) -> erlang:error({lawspec, {contract_failed, Context}});
require(_, Context) -> erlang:error({lawspec, {non_boolean_contract, Context}}).

assert_equal(Actual, Expected, Label) ->
    case lawspec_beam_scalar:equal(Actual, Expected) of
        true ->
            true;
        false ->
            erlang:error(
                {lawspec, {equation_failed, Label, {actual, Actual}, {expected, Expected}}}
            )
    end.

contextual(Identity, Body) ->
    try
        Body()
    catch
        error:{lawspec, Reason}:Stack -> erlang:raise(error, {lawspec, {Identity, Reason}}, Stack)
    end.

helper(<<"incompleteSearch">>, [Reason], _, _) ->
    erlang:error({lawspec, Reason});
helper(<<"unreachable">>, _, _, _) ->
    erlang:error({lawspec, unreachable});
helper(<<"recorded">>, [Key, Value], [_, Type], _) ->
    recorded(Key, Value, Type, none);
helper(<<"recorded">>, [Key, Value], [], _) ->
    recorded_text(Key, lawspec_beam_values:render(Value));
helper(Name, Values, Types, Bits) ->
    lawspec_beam_scalar:helper(Name, Values, Types, Bits).

%% Recorded values use typed portable rendering and the same folder
%% selection as the other targets. Read-only runs never create a recording;
%% only an explicit --update-recorded invocation supplies the update flag.
%% ref:REQ-law-primitives
recorded(Key, Value, Type, Schema) ->
    recorded_text(Key, lawspec_beam_recorded:text(Value, Type, Schema)).

recorded_text(Key, Text) ->
    Path = filename:join([recorded_root() | binary:split(Key, <<"/">>, [global])]),
    case os:getenv("LAWSPEC_UPDATE_RECORDED") of
        "1" ->
            case filelib:ensure_dir(Path) of
                ok ->
                    case file:write_file(Path, <<Text/binary, "\n">>) of
                        ok -> true;
                        {error, Reason} -> recording_io_error(Key, Reason)
                    end;
                {error, Reason} ->
                    recording_io_error(Key, Reason)
            end;
        _ ->
            case file:read_file(Path) of
                {ok, Bytes} ->
                    Stored = trim_recording_newline(Bytes),
                    case Stored =:= Text of
                        true ->
                            true;
                        false ->
                            error(
                                {lawspec,
                                    <<"recorded/", Key/binary, " differs: expected ", Stored/binary,
                                        ", actual ", Text/binary,
                                        " (lawspec test --update-recorded records the new value)">>}
                            )
                    end;
                {error, enoent} ->
                    error(
                        {lawspec,
                            <<"no recording recorded/", Key/binary,
                                "; run lawspec test --update-recorded to record ", Text/binary>>}
                    );
                {error, Reason} ->
                    recording_io_error(Key, Reason)
            end
    end.

trim_recording_newline(<<>>) ->
    <<>>;
trim_recording_newline(Bytes) ->
    case binary:last(Bytes) of
        $\n -> binary:part(Bytes, 0, byte_size(Bytes) - 1);
        _ -> Bytes
    end.

recording_io_error(Key, Reason) ->
    error(
        {lawspec,
            <<"cannot access recording recorded/", Key/binary, ": ",
                (atom_to_binary(Reason, utf8))/binary>>}
    ).

recorded_root() ->
    case os:getenv("LAWSPEC_RECORDED") of
        false -> find_recorded_root();
        "" -> find_recorded_root();
        Given -> unicode:characters_to_binary(Given)
    end.

find_recorded_root() ->
    {ok, Current} = file:get_cwd(),
    Start = unicode:characters_to_binary(Current),
    find_recorded_root(Start, Start).

find_recorded_root(Folder, Start) ->
    case
        filelib:is_regular(filename:join(Folder, <<"lawspec.edn">>)) orelse
            filelib:is_dir(filename:join(Folder, <<"recorded">>))
    of
        true ->
            filename:join(Folder, <<"recorded">>);
        false ->
            case filename:dirname(Folder) of
                Folder -> filename:join(Start, <<"recorded">>);
                Parent -> find_recorded_root(Parent, Start)
            end
    end.

%% BEAM adapters return ordinary native values. An async declaration runs the
%% call in its own process, awaiting its value or original exception. Reuse
%% the parallel group's ownership and context propagation for cancellation.
async_call(Body) -> hd(concurrently([Body])).

%% @doc Each parallel step runs in a monitored worker. Results are returned in
%% source order; exceptions preserve their class and stack. Every sibling
%% finishes before the first exception in source order is raised. Caller
%% cancellation joins the cancelled children. The coordinator keeps
%% messages out of the caller's mailbox. ref:DEC-typed-core-boundary
concurrently(Bodies) ->
    Caller = self(),
    Context = worker_context(),
    {Coordinator, Monitor} = spawn_monitor(fun() ->
        with_worker_context(Context, fun() ->
            process_flag(trap_exit, true),
            ParentMonitor = monitor(process, Caller),
            Workers = [
                {
                    spawn_opt(
                        fun() ->
                            Result =
                                try
                                    {ok, with_worker_context(Context, Body)}
                                catch
                                    Class:Reason:Stack -> {exception, Class, Reason, Stack}
                                end,
                            exit({lawspec_result, Result})
                        end,
                        [link, monitor]
                    ),
                    I
                }
             || {I, Body} <- lists:enumerate(Bodies)
            ],
            Result = collect(Workers, #{}, ParentMonitor),
            exit({lawspec_result, Result})
        end)
    end),
    receive
        {'DOWN', Monitor, process, Coordinator, {lawspec_result, {ok, Results}}} ->
            Results;
        {'DOWN', Monitor, process, Coordinator,
            {lawspec_result, {exception, Class, Reason, Stack}}} ->
            erlang:raise(Class, Reason, Stack);
        {'DOWN', Monitor, process, Coordinator, Reason} ->
            erlang:error({lawspec, {concurrent_group_failed, Reason}})
    end.

collect([], Results, ParentMonitor) ->
    demonitor(ParentMonitor, [flush]),
    Ordered = [maps:get(I, Results) || I <- lists:seq(1, map_size(Results))],
    case [Error || Error = {exception, _, _, _} <- Ordered] of
        [First | _] -> First;
        [] -> {ok, [Value || {ok, Value} <- Ordered]}
    end;
collect(Workers, Results, ParentMonitor) ->
    receive
        {'DOWN', ParentMonitor, process, _, _} ->
            cancel(Workers),
            exit(normal);
        {'DOWN', Monitor, process, Pid, Reason} ->
            case lists:keytake({Pid, Monitor}, 1, Workers) of
                {value, {_, I}, Rest} ->
                    Outcome =
                        case Reason of
                            {lawspec_result, {ok, _} = Value} -> Value;
                            {lawspec_result, {exception, _, _, _} = Error} -> Error;
                            _ -> {exception, error, {lawspec, {concurrent_step_failed, Reason}}, []}
                        end,
                    collect(Rest, Results#{I => Outcome}, ParentMonitor)
            end
    end.

cancel(Workers) ->
    lists:foreach(fun({{Pid, _}, _}) -> exit(Pid, kill) end, Workers),
    lists:foreach(
        fun({{Pid, Monitor}, _}) ->
            receive
                {'DOWN', Monitor, process, Pid, _} -> ok
            end
        end,
        Workers
    ).
