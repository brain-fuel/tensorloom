%% @doc Packet transport ownership is shared by cleartext test nodes and the
%% secure network layer. The receiving process owns registration and sends.
%% ref:DEC-distribution-canonical-wire
-module(lawspec_beam_transport).
-export([open/2, send/3, close/1]).

open(#{module := lawspec_beam_memory_network, network := Network, address := Address}, Receiver) ->
    case lawspec_beam_memory_network:register(Network, Address, Receiver) of
        ok -> {ok, #{module => lawspec_beam_memory_network, pid => Network, address => Address}};
        Error -> Error
    end;
open(#{module := lawspec_beam_socket_transport} = Options, Receiver) ->
    case lawspec_beam_socket_transport:start(Options, Receiver) of
        {ok, Pid} ->
            {ok, #{
                module => lawspec_beam_socket_transport,
                pid => Pid,
                address => lawspec_beam_socket_transport:address(Pid)
            }};
        Error ->
            Error
    end;
open(_, _) ->
    {error, unsupported_transport}.
send(#{module := lawspec_beam_network, pid := Layer}, Peer, Bytes) ->
    try
        lawspec_beam_network:send(Layer, Peer, Bytes)
    catch
        exit:_ -> {error, transport_closed}
    end;
send(#{module := lawspec_beam_memory_network, pid := Network, address := Address}, Peer, Bytes) ->
    try
        lawspec_beam_memory_network:send(Network, Address, Peer, Bytes)
    catch
        exit:_ -> {error, transport_closed}
    end;
send(#{module := lawspec_beam_socket_transport, pid := Pid}, Peer, Bytes) ->
    try
        lawspec_beam_socket_transport:send(Pid, Peer, Bytes)
    catch
        exit:_ -> {error, transport_closed}
    end.
close(#{module := lawspec_beam_network, pid := Layer}) ->
    lawspec_beam_network:stop(Layer);
close(#{module := lawspec_beam_socket_transport, pid := Pid}) ->
    lawspec_beam_socket_transport:stop(Pid);
close(#{module := lawspec_beam_memory_network, pid := Network, address := Address}) ->
    try
        lawspec_beam_memory_network:unregister(Network, Address)
    catch
        exit:_ -> ok
    end.
