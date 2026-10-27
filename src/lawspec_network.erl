%% @doc Native node and transport constructors. Nodes and memory networks
%% follow their creating process; scoped constructors also join on return.
%% Identity and trust options use the shared lawspec-network.conf defaults.
%% ref:DEC-distribution-canonical-wire ref:DEC-idiomatic-generated-types
-module(lawspec_network).
-export([
    options/0,
    with_identity/2,
    with_trusted/2,
    trust_on_first_use/1,
    new_identity/0,
    identity_from_seed/1,
    public_key/1,
    fingerprint/1,
    tcp/2,
    http/2,
    advertise/2,
    memory/1,
    with_memory/2,
    close_memory/1,
    memory_transport/2,
    insecure_memory_transport_for_tests/2,
    partition/2,
    heal/1,
    recorded/1,
    open/1, open/2,
    with_node/2, with_node/3,
    close/1,
    address/1,
    node_fingerprint/1
]).
-export_type([node_handle/0, memory/0, transport/0, identity/0, options/0]).
-opaque node_handle() :: pid().
-opaque memory() :: pid().
-opaque transport() :: map().
-opaque identity() :: {lawspec_node_identity, binary(), binary()}.
-opaque options() :: map().

-spec options() -> options().
options() -> #{}.
-spec with_identity(options(), identity()) -> options().
with_identity(Options, Identity) ->
    _ = public_key(Identity),
    Options#{identity => Identity}.
-spec with_trusted(options(), [binary()]) -> options().
with_trusted(Options, Fingerprints) when is_list(Fingerprints) -> Options#{trusted => Fingerprints}.
-spec trust_on_first_use(options()) -> options().
trust_on_first_use(Options) -> Options#{trusted => none}.
-spec new_identity() -> identity().
new_identity() -> identity_from_seed(crypto:strong_rand_bytes(32)).
-spec identity_from_seed(binary()) -> identity().
identity_from_seed(Seed) ->
    require_crypto(),
    lawspec_beam_network_crypto:identity(Seed).
-spec public_key(identity()) -> binary().
public_key(Identity) ->
    require_crypto(),
    lawspec_beam_network_crypto:public(Identity).
-spec fingerprint(identity()) -> binary().
fingerprint(Identity) ->
    require_crypto(),
    lawspec_beam_network_crypto:fingerprint(Identity).

-spec tcp(binary(), 0..65535) -> transport().
tcp(Host, Port) ->
    #{module => lawspec_beam_socket_transport, kind => tcp, host => Host, port => Port}.
-spec http(binary(), 0..65535) -> transport().
http(Host, Port) ->
    #{module => lawspec_beam_socket_transport, kind => http, host => Host, port => Port}.
-spec advertise(transport(), binary()) -> transport().
advertise(Transport = #{module := lawspec_beam_socket_transport}, Host) ->
    Transport#{advertise_host => Host}.
-spec memory(map() | {memory_options, integer(), float(), float(), float(), boolean()}) -> memory().
memory({memory_options, Seed, Loss, Duplicate, Delay, Record}) ->
    memory(#{seed => Seed, loss => Loss, duplicate => Duplicate, delay => Delay, record => Record});
memory(Options) ->
    unwrap(lawspec_beam_memory_network:start(Options)).
-spec with_memory(map() | tuple(), fun((memory()) -> A)) -> A.
with_memory(Options, Body) ->
    Network = memory(Options),
    try
        Body(Network)
    after
        close_memory(Network)
    end.
-spec close_memory(memory()) -> ok.
close_memory(Network) -> lawspec_beam_memory_network:stop(Network).
-spec memory_transport(memory(), binary()) -> transport().
memory_transport(Network, Name) -> lawspec_beam_memory_network:transport(Network, Name).
-spec insecure_memory_transport_for_tests(memory(), binary()) -> transport().
insecure_memory_transport_for_tests(Network, Name) ->
    lawspec_beam_memory_network:insecure_transport_for_tests(Network, Name).
-spec partition(memory(), [[binary()]]) -> ok.
partition(Network, Groups) -> lawspec_beam_memory_network:partition(Network, Groups).
-spec heal(memory()) -> ok.
heal(Network) -> lawspec_beam_memory_network:heal(Network).
-spec recorded(memory()) -> [binary()].
recorded(Network) -> lawspec_beam_memory_network:recorded(Network).

-spec open(transport()) -> node_handle().
open(Transport) -> open(Transport, options()).
-spec open(transport(), options()) -> node_handle().
open(Transport, Options) -> unwrap(lawspec_beam_node:start_owned(Transport, Options)).
-spec with_node(transport(), fun((node_handle()) -> A)) -> A.
with_node(Transport, Body) -> with_node(Transport, options(), Body).
-spec with_node(transport(), options(), fun((node_handle()) -> A)) -> A.
with_node(Transport, Options, Body) ->
    Node = open(Transport, Options),
    try
        Body(Node)
    after
        close(Node)
    end.
-spec close(node_handle()) -> ok.
close(Node) -> lawspec_beam_node:stop(Node).
-spec address(node_handle()) -> binary().
address(Node) -> lawspec_beam_node:address(Node).
-spec node_fingerprint(node_handle()) -> binary().
node_fingerprint(Node) -> lawspec_beam_node:identity(Node).
unwrap({ok, Value}) -> Value;
unwrap({error, Reason}) -> error({lawspec, {network, Reason}}).
require_crypto() ->
    case code:ensure_loaded(lawspec_beam_network_crypto) of
        {module, lawspec_beam_network_crypto} -> ok;
        _ -> error({lawspec, {network, secure_network_not_available}})
    end.
