# ExUnit hosts one named result per native case. Its tests within a module
# are serial, so scheduled units execute selected case closures on a bounded
# process pool and return each original result to its native ExUnit test.
# ref:REQ-harness-units ref:DEC-native-property-frameworks
defmodule LawSpec.Beam.ExUnitSchedule do
  @moduledoc false
  use GenServer

  def setup(context, unit, random, parallel) do
    config = ExUnit.configuration()
    collection = context.module.__ex_unit__().tests
    {include, exclude} = ExUnit.Filters.normalize(config[:include], config[:exclude])
    ids = config[:only_test_ids]

    selected =
      Enum.filter(collection, fn test ->
        tags =
          Map.merge(test.tags, %{
            test: test.name,
            module: test.module,
            async: false,
            test_group: nil
          })

        (is_nil(ids) or MapSet.member?(ids, {test.module, test.name})) and
          ExUnit.Filters.eval(include, exclude, tags, collection) == :ok
      end)

    groups =
      selected
      |> Enum.sort_by(& &1.tags.lawspec_order)
      |> Enum.chunk_by(& &1.tags.lawspec)

    ordered = :lawspec_beam_schedule.order(unit, random, groups) |> List.flatten()

    timeout =
      if config[:trace] or config[:timeout] == :infinity do
        :infinity
      else
        Enum.reduce(selected, config[:timeout], fn test, allowance ->
          max(allowance, Map.get(test.tags, :lawspec_timeout, config[:timeout]))
        end)
      end

    jobs =
      Enum.map(ordered, fn test ->
        {owner, factory, case_at, index} = test.tags.lawspec_case

        {test.name,
         fn ->
           {_, run} = :lawspec_beam_finite.lookup(owner, case_at, factory, index)
           run.()
         end}
      end)

    limit = if parallel, do: max(2, System.schedulers_online()), else: 1
    {:ok, scheduler} = GenServer.start(__MODULE__, {self(), unit, parallel, limit, timeout, jobs})
    ExUnit.Callbacks.on_exit(fn -> close(scheduler) end)
    {:ok, lawspec_schedule: scheduler}
  end

  def await(scheduler, name) do
    case GenServer.call(scheduler, {:await, name}, :infinity) do
      {:ok, value} -> value
      {:error, kind, reason, stack} -> :erlang.raise(kind, reason, stack)
    end
  end

  def close(scheduler) do
    monitor = Process.monitor(scheduler)

    try do
      result = GenServer.call(scheduler, :close, :infinity)

      receive do
        {:DOWN, ^monitor, :process, ^scheduler, _} -> :ok
      end

      case result do
        :ok -> :ok
        {:error, kind, reason, stack} -> :erlang.raise(kind, reason, stack)
      end
    catch
      :exit, {:noproc, _} -> :ok
      :exit, {:normal, _} -> :ok
    after
      Process.demonitor(monitor, [:flush])
    end
  end

  def init({owner, unit, parallel, limit, timeout, jobs}) do
    {:ok,
     %{
       owner: Process.monitor(owner),
       unit: unit,
       parallel: parallel,
       limit: limit,
       timeout: timeout,
       jobs: jobs,
       active: %{},
       results: %{},
       keys: MapSet.new(Enum.map(jobs, &elem(&1, 0))),
       waiters: %{},
       peak: 0,
       closing: false,
       close_waiters: [],
       reported: false,
       report_result: :ok
     }}
  end

  def handle_call(:close, from, state) do
    finish_close(begin_close(%{state | close_waiters: [from | state.close_waiters]}))
  end

  def handle_call({:await, _}, _, %{closing: true} = state), do: {:reply, closed(), state}

  def handle_call({:await, name}, from, state) do
    if not MapSet.member?(state.keys, name), do: raise("unselected LawSpec case: #{name}")

    case Map.fetch(state.results, name) do
      {:ok, result} ->
        {:reply, result, state}

      :error ->
        waiters = Map.update(state.waiters, name, [from], &[from | &1])
        {:noreply, dispatch(%{state | waiters: waiters})}
    end
  end

  def handle_info({:DOWN, owner, :process, _, _}, %{owner: owner} = state) do
    finish_close(begin_close(state))
  end

  def handle_info({:DOWN, ref, :process, _, reason}, state) do
    case Map.pop(state.active, ref) do
      {nil, _} ->
        {:noreply, state}

      {{name, _pid, timer, timed_out, scope}, active} ->
        if timer != nil, do: Process.cancel_timer(timer)
        # A killed case can leave an async coordinator's DOWN notification in
        # flight. Join its entire task scope before exposing the final result.
        :lawspec_beam_tasks.close(scope)

        result =
          cond do
            timed_out != nil ->
              {:error, :error,
               ExUnit.TimeoutError.exception(timeout: state.timeout, type: "test"), timed_out}

            match?({:lawspec_result, _}, reason) ->
              elem(reason, 1)

            true ->
              {:error, :exit, reason, []}
          end

        next = %{state | active: active} |> finish_report()

        result =
          case {result, next.report_result} do
            {{:ok, _}, {:error, _, _, _} = failure} -> failure
            _ -> result
          end

        Enum.each(Map.get(next.waiters, name, []), &GenServer.reply(&1, result))
        # Deliver the native result before considering more work. Once ExUnit
        # stops asking (max_failures), an empty waiter set cannot refill the pool.
        next = %{
          next
          | waiters: Map.delete(next.waiters, name),
            results: Map.put(next.results, name, result)
        }

        finish_close(dispatch(next))
    end
  end

  def handle_info({:timeout, ref}, state) do
    case Map.fetch(state.active, ref) do
      {:ok, {name, pid, timer, _, scope}} ->
        stack =
          case Process.info(pid, :current_stacktrace) do
            {:current_stacktrace, frames} -> frames
            nil -> []
          end

        Process.exit(pid, :kill)

        {:noreply,
         %{state | active: Map.put(state.active, ref, {name, pid, timer, stack, scope})}}

      :error ->
        {:noreply, state}
    end
  end

  defp dispatch(%{closing: true} = state), do: state
  defp dispatch(%{waiters: waiters} = state) when map_size(waiters) == 0, do: state
  defp dispatch(%{jobs: []} = state), do: state
  defp dispatch(state) when map_size(state.active) >= state.limit, do: state

  defp dispatch(%{jobs: [{name, run} | rest]} = state) do
    scope = :lawspec_beam_tasks.open()

    {pid, ref} =
      spawn_monitor(fn ->
        result =
          try do
            {:ok,
             :lawspec_beam_runtime.with_worker_context(
               [{{:lawspec_beam_tasks, :scopes}, [scope]}],
               run
             )}
          catch
            kind, reason -> {:error, kind, reason, __STACKTRACE__}
          end

        exit({:lawspec_result, result})
      end)

    timer =
      if state.timeout == :infinity,
        do: nil,
        else: Process.send_after(self(), {:timeout, ref}, state.timeout)

    active = Map.put(state.active, ref, {name, pid, timer, nil, scope})
    live = Enum.count(active, fn {_, {_, process, _, _, _}} -> Process.alive?(process) end)
    dispatch(%{state | jobs: rest, active: active, peak: max(state.peak, live)})
  end

  defp finish_report(%{reported: false, active: active, parallel: true} = state)
       when map_size(active) == 0 do
    if state.closing or state.jobs == [] do
      result =
        try do
          :lawspec_beam_schedule.report(state.unit, state.peak)
          :ok
        catch
          kind, reason ->
            IO.puts(:stderr, "LawSpec parallel statistics failed: #{inspect(reason)}")
            {:error, kind, reason, __STACKTRACE__}
        end

      %{state | reported: true, report_result: result}
    else
      state
    end
  end

  defp finish_report(state), do: state

  # Fail-fast ends native discovery, but already running cases keep their own
  # processes until they finish (or reach their existing per-case timeout).
  # Generated resource owners finish cleanup before returning each result.
  defp begin_close(state) do
    running = MapSet.new(state.active, fn {_, {name, _, _, _, _}} -> name end)

    {started, queued} =
      Enum.split_with(state.waiters, fn {name, _} -> MapSet.member?(running, name) end)

    Enum.each(queued, fn {_, waiters} -> Enum.each(waiters, &GenServer.reply(&1, closed())) end)
    %{state | closing: true, jobs: [], waiters: Map.new(started)}
  end

  defp finish_close(%{closing: true, active: active} = state) when map_size(active) == 0 do
    next = finish_report(state)
    Enum.each(next.close_waiters, &GenServer.reply(&1, next.report_result))
    {:stop, :normal, next}
  end

  defp finish_close(state), do: {:noreply, state}

  defp closed, do: {:error, :exit, :lawspec_schedule_closed, []}

  def terminate(_, state) do
    Enum.each(state.active, fn {_, {_, _, timer, _, scope}} ->
      if timer != nil, do: Process.cancel_timer(timer)
      :lawspec_beam_tasks.close(scope)
    end)

    :ok
  end
end
