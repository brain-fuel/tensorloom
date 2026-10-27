%% @doc Native completion events for the CLI. Incomplete or stale reports
%% cannot prove a law ran. Both native frameworks use one writer per run.
%% ref:DEC-never-pass-vacuously ref:DEC-native-property-frameworks
-module(lawspec_beam_report).
-behaviour(eunit_listener).
-export([start/1, init/1, handle_begin/3, handle_end/3, handle_cancel/3, terminate/2]).
-export([open/0, event/2, finish/1]).

open() ->
    case os:getenv("LAWSPEC_BEAM_REPORT") of
        false ->
            none;
        Path ->
            Token =
                case os:getenv("LAWSPEC_BEAM_RUN") of
                    false -> <<>>;
                    T -> text(T)
                end,
            ok = filelib:ensure_dir(Path),
            ok = file:write_file(Path, <<>>),
            State = {Path, Token},
            event(State, #{event => <<"start">>}),
            State
    end.

event(none, _) ->
    ok;
event({Path, Token}, Row) ->
    ok = file:write_file(Path, [json:encode(Row#{run => Token}), "\n"], [append]).

finish(State) -> event(State, #{event => <<"end">>}).

start(Options) -> eunit_listener:start(?MODULE, Options).
init(_) ->
    receive
        {start, _} -> #{report => open(), groups => #{}}
    end.

handle_begin(group, Data, State = #{groups := Groups}) ->
    State#{groups := Groups#{proplists:get_value(id, Data) => proplists:get_value(desc, Data)}};
handle_begin(test, _, State) ->
    State.

handle_end(test, Data, State) ->
    Status =
        case proplists:get_value(status, Data) of
            ok -> <<"passed">>;
            _ -> <<"failed">>
        end,
    report(Data, Status, State),
    State;
handle_end(group, _, State) ->
    State.

handle_cancel(_, Data, State) ->
    report(Data, <<"failed">>, State),
    State.

terminate({ok, _}, #{report := Report}) -> finish(Report);
terminate({error, _}, _) -> ok.

report(Data, Status, #{report := Report, groups := Groups}) ->
    Id = proplists:get_value(id, Data),
    Identity =
        case
            lists:reverse(
                lists:sort([
                    {length(Parent), GroupName}
                 || {Parent, Description} <- maps:to_list(Groups),
                    lists:prefix(Parent, Id),
                    <<"lawspec:", GroupName/binary>> <- [text(Description)]
                ])
            )
        of
            [{_, FoundIdentity} | _] -> FoundIdentity;
            [] -> <<>>
        end,
    {Module, Function, _} =
        case proplists:get_value(source, Data) of
            {_, _, _} = Source -> Source;
            _ -> {undefined, undefined, 0}
        end,
    TestName =
        case proplists:get_value(desc, Data) of
            undefined -> text(io_lib:format("~tp:~tp", [Module, Function]));
            TestDescription -> text(TestDescription)
        end,
    Milliseconds =
        case proplists:get_value(time, Data) of
            T when is_number(T) -> T;
            _ -> 0
        end,
    Row = #{
        event => <<"test">>,
        name => TestName,
        classname => text(Module),
        identity => Identity,
        status => Status,
        time => Milliseconds / 1000
    },
    event(
        Report,
        case Status of
            <<"failed">> ->
                Row#{
                    failure => text(
                        io_lib:format("~tp", [
                            proplists:get_value(
                                status,
                                Data,
                                proplists:get_value(reason, Data)
                            )
                        ])
                    )
                };
            _ ->
                Row
        end
    ).

text(undefined) -> <<>>;
text(Atom) when is_atom(Atom) -> atom_to_binary(Atom);
text(Value) -> unicode:characters_to_binary(Value).
