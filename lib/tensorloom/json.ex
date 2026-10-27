defmodule Tensorloom.JSON do
  @moduledoc false
  def decode(s), do: :json.decode(s) |> from_json()
  def encode(x), do: :json.encode(to_json(x))
  defp from_json(:null), do: nil
  defp from_json(x) when is_list(x), do: Enum.map(x, &from_json/1)
  defp from_json(x) when is_map(x), do: Map.new(x, fn {k, v} -> {k, from_json(v)} end)
  defp from_json(x), do: x
  defp to_json(nil), do: :null
  defp to_json(x) when is_list(x), do: Enum.map(x, &to_json/1)
  defp to_json(x) when is_map(x), do: Map.new(x, fn {k, v} -> {k, to_json(v)} end)
  defp to_json(x), do: x
end
