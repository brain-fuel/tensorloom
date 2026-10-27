case System.get_env("LAWSPEC_SEED") do
  nil -> ExUnit.start()
  seed -> ExUnit.start(seed: String.to_integer(seed))
end

if Code.ensure_loaded?(LawSpec.Beam.ExUnitFormatter) do
  ExUnit.configure(
    formatters: Enum.uniq(ExUnit.configuration()[:formatters] ++ [LawSpec.Beam.ExUnitFormatter])
  )
end

if Code.ensure_loaded?(:lawspec_beam_resources) do
  :lawspec_beam_resources.configure_suite(Process.whereis(ExUnit.Server))
  ExUnit.after_suite(fn _ -> :lawspec_beam_resources.stop_suite() end)
end
