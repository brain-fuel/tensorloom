# Observe ExUnit's real skip events; excluded tests are not reported as run.
# ref:REQ-harness-units ref:DEC-native-property-frameworks
defmodule LawSpec.Beam.ExUnitFormatter do
  @moduledoc false
  use GenServer

  def init(_options), do: {:ok, :lawspec_beam_report.open()}

  def handle_cast(
        {:test_finished, %{state: {:skipped, _}, tags: %{lawspec_skip: {law, reason}}}},
        state
      ) do
    :lawspec_beam_harness.skip(law, reason)
    {:noreply, state}
  end

  def handle_cast({:test_finished, %{state: {:excluded, _}}}, state), do: {:noreply, state}
  def handle_cast({:test_finished, %{state: {:skipped, _}}}, state), do: {:noreply, state}

  def handle_cast({:test_finished, test}, state) do
    :lawspec_beam_report.event(state, %{
      event: "test",
      name: Atom.to_string(test.name),
      classname: inspect(test.module),
      identity: Map.get(test.tags, :lawspec_identity, ""),
      status: if(test.state == nil, do: "passed", else: "failed"),
      time: (test.time || 0) / 1_000_000,
      failure:
        if(test.state == nil,
          do: "",
          else: inspect(test.state, limit: :infinity, printable_limit: :infinity)
        )
    })

    {:noreply, state}
  end

  def handle_cast({:suite_finished, _}, state) do
    :lawspec_beam_report.finish(state)
    {:noreply, state}
  end

  def handle_cast(_event, state), do: {:noreply, state}
end
