%% @doc A cancellation scope owns every worker started beneath it, including
%% nested async calls and parallel groups. Registration precedes application
%% code, so closing a scope cannot race a new child's first side effect.
%% ref:DEC-async-native-tasks ref:DEC-typed-core-boundary
-module(lawspec_beam_tasks).
-behaviour(gen_server).
-export([open/0, with_scope/1, attach/1, adopt/2, close/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2]).

%% A persistent service can own a scope without installing it in its own
%% process dictionary. Its workers attach before running application code.
open() ->
    {ok, Scope} = gen_server:start(?MODULE, self(), []),
    Scope.

with_scope(Body) ->
    Scope = open(),
    Key = {?MODULE, scopes},
    Previous = get(Key),
    Parents =
        case Previous of
            undefined -> [];
            _ -> Previous
        end,
    put(Key, [Scope | Parents]),
    try
        Body(Scope)
    after
        case Previous of
            undefined -> erase(Key);
            _ -> put(Key, Previous)
        end,
        close(Scope)
    end.

%% A nested worker registers with every enclosing scope. An outer timeout
%% therefore joins its grandchildren even if their coordinator is blocked.
attach(Scopes) ->
    lists:foreach(
        fun(Scope) ->
            ok = gen_server:call(Scope, {attach, self()}, infinity)
        end,
        Scopes
    ).

%% A service owner may also attach a child before exposing its handle.
adopt(Scope, Worker) -> gen_server:call(Scope, {adopt, Worker}, infinity).

close(Scope) ->
    try
        gen_server:call(Scope, close, infinity)
    catch
        exit:{noproc, _} -> ok;
        exit:{normal, _} -> ok
    end.

init(Owner) ->
    process_flag(trap_exit, true),
    {ok, #{
        owner => Owner,
        owner_monitor => monitor(process, Owner),
        workers => #{},
        monitors => #{},
        closing => false,
        waiters => []
    }}.

handle_call({adopt, Pid}, {Owner, _}, State = #{owner := Owner}) ->
    handle_call({attach, Pid}, none, State);
handle_call({adopt, _}, _, State) ->
    {reply, {error, not_scope_owner}, State};
handle_call({attach, Owner}, _, State = #{owner := Owner, closing := true}) ->
    {reply, {error, closed}, State};
handle_call({attach, Owner}, _, State = #{owner := Owner}) ->
    {reply, ok, State};
handle_call({attach, Pid}, _, State = #{closing := true}) ->
    exit(Pid, kill),
    {reply, {error, closed}, State};
handle_call({attach, Pid}, _, State = #{workers := Workers, monitors := Monitors}) ->
    case maps:is_key(Pid, Workers) of
        true ->
            {reply, ok, State};
        false ->
            Monitor = monitor(process, Pid),
            link(Pid),
            {reply, ok, State#{
                workers := Workers#{Pid => Monitor}, monitors := Monitors#{Monitor => Pid}
            }}
    end;
handle_call(close, From, State = #{waiters := Waiters}) ->
    finish(begin_close(State#{waiters := [From | Waiters]})).

handle_cast(_, State) -> {noreply, State}.

handle_info({'DOWN', Owner, process, _, _}, State = #{owner_monitor := Owner}) ->
    finish(begin_close(State));
handle_info({'DOWN', Monitor, process, _, _}, State = #{workers := Workers, monitors := Monitors}) ->
    case maps:take(Monitor, Monitors) of
        error -> {noreply, State};
        {Pid, Rest} -> finish(State#{workers := maps:remove(Pid, Workers), monitors := Rest})
    end;
handle_info(_, State) ->
    {noreply, State}.

begin_close(State = #{workers := Workers}) ->
    maps:foreach(fun(Pid, _) -> exit(Pid, kill) end, Workers),
    State#{closing := true}.

finish(State = #{closing := true, workers := Workers, waiters := Waiters}) when
    map_size(Workers) =:= 0
->
    lists:foreach(fun(From) -> gen_server:reply(From, ok) end, Waiters),
    {stop, normal, State};
finish(State) ->
    {noreply, State}.
