defmodule Tensorloom.Compiler do
  @moduledoc "Emits nn.Module source or saves an inference export (ref:DEC-compiler)."
  def compile(g, opts) do
    mode = Keyword.get(opts, :lowering, :expanded) |> to_string()
    profile = Keyword.get(opts, :profile, :eager) |> to_string()

    cond do
      mode not in ["native", "expanded"] ->
        {:error, [%{op: "compile", code: "unsupported-lowering", message: mode}]}

      profile not in ["eager", "export"] ->
        {:error, [%{op: "compile", code: "unsupported-profile", message: profile}]}

      profile == "export" and
          (!Keyword.has_key?(opts, :example_inputs) or !Keyword.has_key?(opts, :path)) ->
        {:error,
         [
           %{
             op: "compile",
             code: "export-options",
             message: "export requires example_inputs (JSON values) and path"
           }
         ]}

      true ->
        Tensorloom.protect(fn -> generate(g, opts, mode, profile) end)
    end
  end

  defp generate(g, opts, mode, profile) do
    root = Path.expand("../..", __DIR__)
    python = Keyword.get(opts, :python, Path.join(root, ".venv/bin/python"))
    payload = %{"graph" => g, "lowering" => mode, "profile" => profile}

    payload =
      if profile == "export",
        do:
          Map.merge(payload, %{
            "examples" => Keyword.fetch!(opts, :example_inputs),
            "path" => Path.expand(Keyword.fetch!(opts, :path)),
            "dynamic_dims" => Keyword.get(opts, :dynamic_dims, %{})
          }),
        else: payload

    request = IO.iodata_to_binary(Tensorloom.JSON.encode(payload))

    path =
      Path.join(System.tmp_dir!(), "tensorloom-#{System.unique_integer([:positive])}.json")

    try do
      File.write!(path, request)
      action = if profile == "export", do: "export", else: "compile"

      case System.cmd(python, ["-m", "tensorloom.cli", action, path],
             cd: root,
             env: [{"PYTHONPATH", Path.join(root, "python")}],
             stderr_to_stdout: true
           ) do
        {out, 0} -> {:ok, if(profile == "export", do: Tensorloom.JSON.decode(out), else: out)}
        {out, _} -> {:error, [%{op: "compile", code: "backend-error", message: out}]}
      end
    after
      File.rm(path)
    end
  end
end
