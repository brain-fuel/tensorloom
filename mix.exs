defmodule Tensorloom.MixProject do
  use Mix.Project

  def project do
    [
      app: :tensorloom,
      version: "0.2.0",
      elixir: "~> 1.20.0",
      erlc_paths: ["generated/elixir"] ++ test_paths(),
      elixirc_paths: ["lib", "generated/elixir"] ++ test_paths(),
      deps: [{:stream_data, "== 1.4.0", only: :test}]
    ]
  end

  defp test_paths, do: if(Mix.env() == :test, do: ["test/support"], else: [])

  def application do
    [extra_applications: [:logger]]
  end
end
