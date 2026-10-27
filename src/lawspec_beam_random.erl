%% @doc The portable SplitMix64 sequence, with its state passed explicitly so
%% concurrent callers never draw from one another's stream.
%% ref:splitmix ref:DEC-portable-seeded-generation
-module(lawspec_beam_random).
-export([seed/1, next/1, below/2]).
-define(MASK, 16#ffffffffffffffff).

seed(N) when is_integer(N) -> N band ?MASK.
next(State) ->
    S = (State + 16#9e3779b97f4a7c15) band ?MASK,
    A = ((S bxor (S bsr 30)) * 16#bf58476d1ce4e5b9) band ?MASK,
    B = ((A bxor (A bsr 27)) * 16#94d049bb133111eb) band ?MASK,
    {B bxor (B bsr 31), S}.
below(Bound, State) when is_integer(Bound), Bound > 0 ->
    {N, Next} = next(State),
    {N rem Bound, Next};
below(Bound, State) when is_integer(Bound), Bound =< 0 -> {0, State}.
