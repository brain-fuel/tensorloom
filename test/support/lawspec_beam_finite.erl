%% @doc Ordered finite covers and case lookup. Literal factories keep symbol
%% scopes inside each attempt; shared suffixes and integer ranges stay compact.
%% ref:DEC-planned-generation ref:DEC-native-property-frameworks
-module(lawspec_beam_finite).
-export([
    cases/2,
    with_timeout/2,
    lookup/4,
    plan_cases/1,
    case_at/2,
    fold/3,
    count/1,
    values/2,
    description/2,
    name/2
]).

cases(Cover, Make) ->
    lists:reverse(
        fold(Cover, [], fun(Index, Factories, Acc) ->
            {continue, [Make(Index, Factories) | Acc]}
        end)
    ).

with_timeout(Seconds, Cases) -> [{timeout, Seconds, Case} || Case <- Cases].

plan_cases(Plan) ->
    lists:append([
        case Group of
            {fixed, Entries} -> Entries;
            {finite, Cover, Make} -> cases(Cover, Make)
        end
     || Group <- Plan
    ]).

lookup(Module, At, Factory, Index) ->
    case erlang:function_exported(Module, At, 1) of
        true -> Module:At(Index);
        false -> lists:nth(Index + 1, Module:Factory())
    end.

case_at(Plan, Index) when is_integer(Index), Index >= 0 ->
    group_at(Plan, Index);
case_at(_, _) ->
    error({lawspec, invalid_finite_case_index}).

group_at([], _) ->
    error({lawspec, finite_case_index_out_of_range});
group_at([{fixed, Entries} | Rest], Index) ->
    Size = length(Entries),
    case Index < Size of
        true -> lists:nth(Index + 1, Entries);
        false -> group_at(Rest, Index - Size)
    end;
group_at([{finite, Cover, Make} | Rest], Index) ->
    Size = count(Cover),
    case Index < Size of
        true -> Make(Index + 1, cover_at(Cover, Index, []));
        false -> group_at(Rest, Index - Size)
    end.

cover_at({copies, Count}, Index, Prefix) when Index < Count -> lists:reverse(Prefix);
cover_at({branches, Branches}, Index, Prefix) -> branch_at(Branches, Index, Prefix).

branch_at([{Domain, Suffix} | Rest], Index, Prefix) ->
    SuffixCount = count(Suffix),
    Size = domain_count(Domain) * SuffixCount,
    case Index < Size of
        true ->
            Factory = domain_at(Domain, Index div SuffixCount),
            cover_at(Suffix, Index rem SuffixCount, [Factory | Prefix]);
        false ->
            branch_at(Rest, Index - Size, Prefix)
    end.

domain_at({values, Factories}, Index) ->
    lists:nth(Index + 1, Factories);
domain_at({integers, First, _}, Index) ->
    Value = First + Index,
    fun(_) -> Value end.

count({copies, Count}) when is_integer(Count), Count >= 0 -> Count;
count({branches, Branches}) ->
    lists:sum([domain_count(Domain) * count(Suffix) || {Domain, Suffix} <- Branches]).

domain_count({values, Factories}) ->
    length(Factories);
domain_count({integers, First, Last}) when is_integer(First), is_integer(Last) ->
    max(0, Last - First + 1).

fold(Cover, Initial, Visit) ->
    {_, _, Result} = walk(Cover, [], 1, Initial, Visit),
    Result.

walk({copies, Count}, Prefix, Index, Acc, Visit) when is_integer(Count), Count >= 0 ->
    copies(Count, lists:reverse(Prefix), Index, Acc, Visit);
walk({branches, Branches}, Prefix, Index, Acc, Visit) ->
    branches(Branches, Prefix, Index, Acc, Visit).

copies(0, _, Index, Acc, _) ->
    {continue, Index, Acc};
copies(Count, Factories, Index, Acc, Visit) ->
    case Visit(Index, Factories, Acc) of
        {halt, Next} -> {halt, Index, Next};
        {continue, Next} -> copies(Count - 1, Factories, Index + 1, Next, Visit)
    end.

branches([], _, Index, Acc, _) ->
    {continue, Index, Acc};
branches([{Domain, Suffix} | Rest], Prefix, Index, Acc, Visit) ->
    case domain(Domain, Suffix, Prefix, Index, Acc, Visit) of
        {halt, _, _} = Result -> Result;
        {continue, NextIndex, Next} -> branches(Rest, Prefix, NextIndex, Next, Visit)
    end.

domain({values, Factories}, Suffix, Prefix, Index, Acc, Visit) ->
    literals(Factories, Suffix, Prefix, Index, Acc, Visit);
domain({integers, First, Last}, Suffix, Prefix, Index, Acc, Visit) ->
    integers(First, Last, Suffix, Prefix, Index, Acc, Visit).

literals([], _, _, Index, Acc, _) ->
    {continue, Index, Acc};
literals([Factory | Rest], Suffix, Prefix, Index, Acc, Visit) ->
    case walk(Suffix, [Factory | Prefix], Index, Acc, Visit) of
        {halt, _, _} = Result -> Result;
        {continue, NextIndex, Next} -> literals(Rest, Suffix, Prefix, NextIndex, Next, Visit)
    end.

integers(First, Last, _, _, Index, Acc, _) when First > Last -> {continue, Index, Acc};
integers(First, Last, Suffix, Prefix, Index, Acc, Visit) ->
    case walk(Suffix, [fun(_) -> First end | Prefix], Index, Acc, Visit) of
        {halt, _, _} = Result ->
            Result;
        {continue, NextIndex, Next} ->
            integers(First + 1, Last, Suffix, Prefix, NextIndex, Next, Visit)
    end.

values(Symbols, Factories) -> [Factory(Symbols) || Factory <- Factories].

description(Shown, Index) ->
    Suffix = " boundary " ++ integer_to_list(Index),
    lists:sublist(unicode:characters_to_list(Shown), 200 - length(Suffix)) ++ Suffix.

name(Prefix, Index) -> <<Prefix/binary, (integer_to_binary(Index))/binary>>.
