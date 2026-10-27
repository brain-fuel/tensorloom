# Native node constructors. ref:DEC-distribution-canonical-wire ref:DEC-idiomatic-generated-types
defmodule LawSpec.Network do
  @opaque node_handle :: :lawspec_network.node_handle()
  @opaque memory :: :lawspec_network.memory()
  @opaque transport :: :lawspec_network.transport()
  @opaque identity :: :lawspec_network.identity()
  @opaque options :: :lawspec_network.options()

  @spec options() :: options()
  def options, do: :lawspec_network.options()
  @spec with_identity(options(), identity()) :: options()
  defdelegate with_identity(options, identity), to: :lawspec_network
  @spec with_trusted(options(), [String.t()]) :: options()
  defdelegate with_trusted(options, fingerprints), to: :lawspec_network
  @spec trust_on_first_use(options()) :: options()
  defdelegate trust_on_first_use(options), to: :lawspec_network
  @spec new_identity() :: identity()
  defdelegate new_identity(), to: :lawspec_network
  @spec identity_from_seed(binary()) :: identity()
  defdelegate identity_from_seed(seed), to: :lawspec_network
  @spec public_key(identity()) :: binary()
  defdelegate public_key(identity), to: :lawspec_network
  @spec fingerprint(identity()) :: String.t()
  defdelegate fingerprint(identity), to: :lawspec_network

  @spec tcp(String.t(), non_neg_integer()) :: transport()
  def tcp(host \\ "127.0.0.1", port \\ 0), do: :lawspec_network.tcp(host, port)
  @spec http(String.t(), non_neg_integer()) :: transport()
  def http(host \\ "127.0.0.1", port \\ 0), do: :lawspec_network.http(host, port)
  @spec advertise(transport(), String.t()) :: transport()
  defdelegate advertise(transport, host), to: :lawspec_network
  @spec memory(keyword() | map()) :: memory()
  def memory(options \\ []), do: :lawspec_network.memory(Map.new(options))
  @spec with_memory(keyword() | map(), (memory() -> result)) :: result when result: var
  def with_memory(options \\ [], body), do: :lawspec_network.with_memory(Map.new(options), body)
  @spec close_memory(memory()) :: :ok
  defdelegate close_memory(memory), to: :lawspec_network
  @spec memory_transport(memory(), String.t()) :: transport()
  defdelegate memory_transport(memory, name), to: :lawspec_network
  @spec insecure_memory_transport_for_tests(memory(), String.t()) :: transport()
  defdelegate insecure_memory_transport_for_tests(memory, name), to: :lawspec_network
  @spec partition(memory(), [[String.t()]]) :: :ok
  defdelegate partition(memory, groups), to: :lawspec_network
  @spec heal(memory()) :: :ok
  defdelegate heal(memory), to: :lawspec_network
  @spec recorded(memory()) :: [binary()]
  defdelegate recorded(memory), to: :lawspec_network

  @spec open(transport(), options()) :: node_handle()
  def open(transport, options \\ options()), do: :lawspec_network.open(transport, options)
  @spec with_node(transport(), options(), (node_handle() -> result)) :: result when result: var
  def with_node(transport, options \\ options(), body),
    do: :lawspec_network.with_node(transport, options, body)

  @spec close(node_handle()) :: :ok
  defdelegate close(node), to: :lawspec_network
  @spec address(node_handle()) :: String.t()
  defdelegate address(node), to: :lawspec_network
  @spec node_fingerprint(node_handle()) :: String.t()
  defdelegate node_fingerprint(node), to: :lawspec_network
end
