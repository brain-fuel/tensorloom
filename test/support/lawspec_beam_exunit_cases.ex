# A law's checked cases as ExUnit tests: one test per example, boundary and
# property, named for the case, so generated test modules name only their
# laws. The cases themselves are the shared Erlang test module's.
# ref:DEC-native-property-frameworks ref:DEC-readable-output-default
defmodule LawSpec.Beam.ExUnitCases do
  @moduledoc false

  # law(cases_module, :law_name, "unit", timeout_ms) defines the law's
  # tests; a law's order in its unit runs them through the unit's schedule
  # (LawSpec.Beam.ExUnitSchedule).
  defmacro law(module, name, unit, allowance, order \\ nil) do
    quote bind_quoted: [
            module: module,
            name: name,
            unit: unit,
            allowance: allowance,
            order: order
          ] do
      factory = :"#{name}_test_"
      case_at = :"#{name}_case_at"

      for {{label, action}, index} <- Enum.with_index(apply(module, factory, [])) do
        @tag lawspec: Atom.to_string(name)
        @tag lawspec_identity: unit <> "::" <> Atom.to_string(name)
        @tag lawspec_label: List.to_string(label)
        @tag lawspec_timeout: allowance

        if order do
          @tag lawspec_order: {order, index}
          @tag lawspec_case: {module, factory, case_at, index}
          @tag timeout: :infinity
        else
          @tag timeout: allowance
        end

        case action do
          {:skip, law, reason} -> @tag skip: reason, lawspec_skip: {law, reason}
          _ -> :ok
        end

        if order do
          test List.to_string(label), context do
            LawSpec.Beam.ExUnitSchedule.await(context.lawspec_schedule, context.test)
          end
        else
          test List.to_string(label) do
            {_, run} =
              :lawspec_beam_finite.lookup(
                unquote(module),
                unquote(case_at),
                unquote(factory),
                unquote(index)
              )

            run.()
          end
        end
      end
    end
  end
end
