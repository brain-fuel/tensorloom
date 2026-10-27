%% @doc The shared BEAM scalar domain. Exact arithmetic never uses native
%% floating-point numbers, and IEEE values keep their bits because Erlang
%% cannot represent infinity or NaN. Every target facade uses these same
%% operations and checked bridges. ref:DEC-portable-exact-arithmetic
%% ref:ieee-754 ref:DEC-explicit-machine-profile
-module(lawspec_beam_scalar).
-export([
    literal/1, literal/2,
    wire/2,
    validate/3,
    convert/3,
    binary/5,
    add/4,
    subtract/4,
    multiply/4,
    divide/4,
    quotient/4,
    remainder/4,
    power/4,
    less/4,
    less_or_equal/4,
    greater/4,
    greater_or_equal/4,
    not_equal/2,
    negate/1,
    equal/2,
    compare/2,
    helper/4,
    distinct_members/1,
    ratio/2,
    decimal/2,
    exact/1,
    float_bits/2,
    float_from_ratio/4,
    float_from_native/2,
    float_to_native/1,
    float_class/1,
    float_binary/3,
    float_compare/2,
    integer_type/1,
    bounds/2,
    type/1,
    new_symbol/1
]).
-export_type([ieee/0, rational/0, decimal/0, complex/0, symbol/0]).

-opaque ieee() :: {ls_float, 32 | 64, non_neg_integer()}.
-opaque rational() :: {ls_ratio, integer(), pos_integer()}.
-opaque decimal() :: {ls_decimal, integer(), integer()}.
-opaque complex() :: {ls_complex, 32 | 64, ieee(), ieee()}.
-opaque symbol() :: {ls_symbol, term(), binary()}.

%% @doc Literal symbols are interned by the caller's context, so two uses of
%% one literal agree without sharing identities between independent runs.
%% ref:DEC-portable-exact-arithmetic
literal(Value) -> literal(Value, make_ref()).

literal(#{<<"type">> := T} = Value, Context) ->
    case T of
        <<"Bool">> ->
            maps:get(<<"value">>, Value);
        <<"Decimal">> ->
            decimal(
                integer_field(<<"coefficient">>, Value),
                integer_field(<<"exponent">>, Value)
            );
        <<"Rational">> ->
            ratio(
                integer_field(<<"numerator">>, Value),
                integer_field(<<"denominator">>, Value)
            );
        <<"Float32">> ->
            float_bits(32, hex(maps:get(<<"bits">>, Value)));
        <<"Float64">> ->
            float_bits(64, hex(maps:get(<<"bits">>, Value)));
        <<"Complex64">> ->
            {ls_complex, 32, literal(maps:get(<<"real">>, Value), Context),
                literal(maps:get(<<"imaginary">>, Value), Context)};
        <<"Complex128">> ->
            {ls_complex, 64, literal(maps:get(<<"real">>, Value), Context),
                literal(maps:get(<<"imaginary">>, Value), Context)};
        <<"Text">> ->
            text(maps:get(<<"units">>, Value));
        <<"Bytes">> ->
            list_to_binary(checked_units(T, maps:get(<<"units">>, Value)));
        <<"CodePointText">> ->
            {ls_raw, T, checked_units(T, maps:get(<<"units">>, Value))};
        <<"Utf16Text">> ->
            {ls_raw, T, checked_units(T, maps:get(<<"units">>, Value))};
        <<"Char">> ->
            unit(T, maps:get(<<"value">>, Value));
        <<"CodePoint">> ->
            unit(T, maps:get(<<"value">>, Value));
        <<"CodeUnit16">> ->
            unit(T, maps:get(<<"value">>, Value));
        <<"Unit">> ->
            ls_unit;
        <<"Null">> ->
            ls_null;
        <<"Undefined">> ->
            ls_undefined;
        <<"Symbol">> ->
            {ls_symbol, {Context, maps:get(<<"id">>, Value)},
                text(unicode:characters_to_list(maps:get(<<"description">>, Value)))};
        _ when T =:= <<"Nullable">>; T =:= <<"Optional">> ->
            {ls_presence, T,
                case maps:get(<<"value">>, Value) of
                    null -> none;
                    Present -> {some, literal(Present, Context)}
                end};
        _ ->
            require(integer_type(T), {unknown_scalar, T}),
            integer_field(<<"value">>, Value)
    end.

integer_field(Key, Value) ->
    case maps:get(Key, Value) of
        N when is_integer(N) -> N;
        B when is_binary(B) -> binary_to_integer(B)
    end.

hex(B) when is_binary(B) -> binary_to_integer(B, 16).

%% @doc The public scalar wire shape is shared with the Core oracle and other
%% targets. No native external-term format crosses this boundary.
%% ref:DEC-distribution-canonical-wire
wire(Value, T) when is_binary(T) -> wire(Value, type(T));
wire({ls_presence, T, Presence}, {T, Args}) ->
    #{
        <<"type">> => T,
        <<"value">> =>
            case {Presence, Args} of
                {none, _} -> null;
                {{some, V}, [Inner]} -> wire(V, Inner);
                _ -> fail(presence_wire_requires_element_type)
            end
    };
wire(Value, {T, []}) ->
    Base = #{<<"type">> => T},
    case {T, Value} of
        {<<"Bool">>, B} when is_boolean(B) -> Base#{<<"value">> => B};
        {<<"Decimal">>, {ls_decimal, C, E}} ->
            Base#{
                <<"coefficient">> => integer_to_binary(C), <<"exponent">> => integer_to_binary(E)
            };
        {<<"Rational">>, {ls_ratio, N, D}} ->
            Base#{
                <<"numerator">> => integer_to_binary(N), <<"denominator">> => integer_to_binary(D)
            };
        {_, {ls_float, W, Bits}} ->
            Digits = integer_to_binary(Bits, 16),
            Hex = string:lowercase(<<
                (binary:copy(<<"0">>, W div 4 - byte_size(Digits)))/binary, Digits/binary
            >>),
            Base#{<<"bits">> => Hex};
        {_, {ls_complex, W, R, I}} ->
            FT = float_type(W),
            Base#{<<"real">> => wire(R, FT), <<"imaginary">> => wire(I, FT)};
        {<<"Text">>, B} ->
            Base#{<<"units">> => unicode:characters_to_list(B)};
        {<<"Bytes">>, B} ->
            Base#{<<"units">> => binary_to_list(B)};
        {_, {ls_raw, _, Units}} ->
            Base#{<<"units">> => Units};
        {_, {ls_symbol, {_, Id}, Desc}} ->
            Base#{<<"id">> => Id, <<"description">> => Desc};
        {_, N} when is_integer(N) ->
            Base#{
                <<"value">> =>
                    case integer_type(T) of
                        true -> integer_to_binary(N);
                        false -> N
                    end
            };
        {_, A} when A =:= ls_unit; A =:= ls_null; A =:= ls_undefined -> Base;
        _ ->
            fail({invalid_scalar_wire, T})
    end.

%% @doc A bridge checks the specification's width and domain, irrespective of
%% the BEAM word size. Containers keep their presence and constructor tags.
%% ref:DEC-explicit-machine-profile ref:DEC-portable-exact-arithmetic
validate(Value, Name, Bits) when is_binary(Name) ->
    validate(Value, type(Name), Bits);
validate(Value, {Name, []}, Bits) ->
    case {Name, Value} of
        {<<"Bool">>, B} when is_boolean(B) -> B;
        {<<"Text">>, B} when is_binary(B) ->
            require(is_list(unicode:characters_to_list(B)), invalid_unicode),
            B;
        {<<"Bytes">>, B} when is_binary(B) -> B;
        {<<"Char">>, N} ->
            unit(Name, N);
        {<<"CodePoint">>, N} ->
            unit(Name, N);
        {<<"CodeUnit16">>, N} ->
            unit(Name, N);
        {<<"CodePointText">>, {ls_raw, Name, Units}} ->
            {ls_raw, Name, checked_units(Name, Units)};
        {<<"Utf16Text">>, {ls_raw, Name, Units}} ->
            {ls_raw, Name, checked_units(Name, Units)};
        {<<"Decimal">>, {ls_decimal, C, E}} when is_integer(C), is_integer(E) -> Value;
        {<<"Rational">>, {ls_ratio, N, D}} ->
            ratio(N, D);
        {<<"Float32">>, {ls_float, 32, B}} ->
            float_bits(32, B);
        {<<"Float64">>, {ls_float, 64, B}} ->
            float_bits(64, B);
        {<<"Complex64">>, {ls_complex, 32, R, I}} ->
            {ls_complex, 32, validate(R, <<"Float32">>, Bits), validate(I, <<"Float32">>, Bits)};
        {<<"Complex128">>, {ls_complex, 64, R, I}} ->
            {ls_complex, 64, validate(R, <<"Float64">>, Bits), validate(I, <<"Float64">>, Bits)};
        {<<"Symbol">>, {ls_symbol, _, D}} when is_binary(D) -> Value;
        {<<"Unit">>, ls_unit} ->
            Value;
        {<<"Null">>, ls_null} ->
            Value;
        {<<"Undefined">>, ls_undefined} ->
            Value;
        _ ->
            require(integer_type(Name) andalso is_integer(Value), {invalid_value, Name}),
            check_bounds(Value, Name, Bits)
    end;
validate(Value, {<<"List">>, [Element]}, Bits) when is_list(Value) ->
    [validate(V, Element, Bits) || V <- Value];
validate({ls_data, <<"Maybe::Absent">>, []} = V, {<<"Maybe">>, [_]}, _) ->
    V;
validate({ls_data, <<"Maybe::Present">>, [V]}, {<<"Maybe">>, [T]}, Bits) ->
    {ls_data, <<"Maybe::Present">>, [validate(V, T, Bits)]};
validate({ls_data, <<"Either::Left">>, [V]}, {<<"Either">>, [T, _]}, Bits) ->
    {ls_data, <<"Either::Left">>, [validate(V, T, Bits)]};
validate({ls_data, <<"Either::Right">>, [V]}, {<<"Either">>, [_, T]}, Bits) ->
    {ls_data, <<"Either::Right">>, [validate(V, T, Bits)]};
validate({ls_presence, Kind, Presence}, {Kind, [T]}, Bits) when
    Kind =:= <<"Nullable">>; Kind =:= <<"Optional">>
->
    {ls_presence, Kind,
        case Presence of
            none -> none;
            {some, V} -> {some, validate(V, T, Bits)}
        end};
validate(_, T, _) ->
    fail({invalid_value, T}).

convert(V, T, Bits) when is_binary(T) -> convert(V, type(T), Bits);
convert(V, {T, []}, Bits) ->
    case T of
        <<"Rational">> ->
            {N, D} = exact(V),
            ratio(N, D);
        <<"Decimal">> ->
            {N, D} = exact(V),
            finite_decimal(N, D);
        <<"Float32">> ->
            to_float(V, 32);
        <<"Float64">> ->
            to_float(V, 64);
        <<"Complex64">> ->
            to_complex(V, 32);
        <<"Complex128">> ->
            to_complex(V, 64);
        _ ->
            case integer_type(T) of
                true ->
                    {N, D} = exact(V),
                    require(N rem D =:= 0, {fractional_conversion, T}),
                    check_bounds(N div D, T, Bits);
                false ->
                    validate(V, {T, []}, Bits)
            end
    end;
convert(ls_null, {<<"Nullable">>, [_]}, _) ->
    {ls_presence, <<"Nullable">>, none};
convert(ls_undefined, {<<"Optional">>, [_]}, _) ->
    {ls_presence, <<"Optional">>, none};
convert({ls_presence, Kind, P}, {Kind, [T]}, Bits) when
    Kind =:= <<"Nullable">>; Kind =:= <<"Optional">>
->
    {ls_presence, Kind,
        case P of
            none -> none;
            {some, V} -> {some, convert(V, T, Bits)}
        end};
convert(V, T, Bits) ->
    validate(V, T, Bits).

integer_type(T) ->
    lists:member(T, [
        <<"Int8">>,
        <<"Int16">>,
        <<"Int32">>,
        <<"Int64">>,
        <<"UInt8">>,
        <<"UInt16">>,
        <<"UInt32">>,
        <<"UInt64">>,
        <<"IntSize">>,
        <<"UIntSize">>,
        <<"UIntPtr">>,
        <<"Integer">>,
        <<"BigInt">>,
        <<"BigUInt">>
    ]).

bounds(T, Bits) ->
    require(Bits =:= 32 orelse Bits =:= 64, invalid_machine_bits),
    case T of
        <<"Integer">> -> {none, none};
        <<"BigInt">> -> {none, none};
        <<"BigUInt">> -> {0, none};
        <<"IntSize">> -> signed_bounds(Bits);
        <<"UIntSize">> -> {0, (1 bsl Bits) - 1};
        <<"UIntPtr">> -> {0, (1 bsl Bits) - 1};
        <<"Int", W/binary>> -> signed_bounds(binary_to_integer(W));
        <<"UInt", W/binary>> -> {0, (1 bsl binary_to_integer(W)) - 1};
        _ -> fail({unknown_integer, T})
    end.

signed_bounds(W) -> {-(1 bsl (W - 1)), (1 bsl (W - 1)) - 1}.
check_bounds(V, T, Bits) ->
    {Lo, Hi} = bounds(T, Bits),
    require(
        (Lo =:= none orelse V >= Lo) andalso (Hi =:= none orelse V =< Hi),
        {integer_out_of_range, T}
    ),
    V.

%% @doc Type keys are parsed without creating atoms from input. Parentheses
%% preserve the boundaries of nested Either and presence arguments.
%% ref:DEC-portable-exact-arithmetic
type(T) when is_tuple(T) -> T;
type(T) when is_binary(T) ->
    {Parts, Rest} = type_parts(type_tokens(binary_to_list(T), [], []), []),
    require(Rest =:= [], invalid_type_key),
    type_application(Parts).

type_tokens([], Word, Acc) ->
    lists:reverse(flush_word(Word, Acc));
type_tokens([C | Cs], Word, Acc) when C =:= $(; C =:= $) ->
    type_tokens(Cs, [], [C | flush_word(Word, Acc)]);
type_tokens([C | Cs], Word, Acc) when C =:= $\s; C =:= $\t ->
    type_tokens(Cs, [], flush_word(Word, Acc));
type_tokens([C | Cs], Word, Acc) ->
    type_tokens(Cs, [C | Word], Acc).
flush_word([], Acc) -> Acc;
flush_word(Word, Acc) -> [list_to_binary(lists:reverse(Word)) | Acc].
type_parts([], Acc) ->
    {lists:reverse(Acc), []};
type_parts([$) | Rest], Acc) ->
    {lists:reverse(Acc), [$) | Rest]};
type_parts([$( | Rest], Acc) ->
    {Inner, After} = type_parts(Rest, []),
    case After of
        [$) | Next] -> type_parts(Next, [type_application(Inner) | Acc]);
        _ -> fail(invalid_type_key)
    end;
type_parts([Word | Rest], Acc) ->
    type_parts(Rest, [{Word, []} | Acc]).
type_application([{Name, []}]) ->
    {Name, []};
type_application([{Name, []} | Args]) when
    Name =:= <<"List">>;
    Name =:= <<"Maybe">>;
    Name =:= <<"Optional">>;
    Name =:= <<"Nullable">>
->
    {Name, [type_application(Args)]};
type_application([{Name, []} | Args]) ->
    {Name, Args};
type_application([T]) ->
    T;
type_application(_) ->
    fail(invalid_type_key).

%% @doc Fractions are normalized with a positive denominator; decimal
%% conversion succeeds only when its expansion terminates in base ten.
%% ref:decimal-arithmetic
ratio(N, D) when is_integer(N), is_integer(D), D =/= 0 ->
    G = gcd(abs(N), abs(D)),
    S =
        case D < 0 of
            true -> -1;
            false -> 1
        end,
    {ls_ratio, S * N div G, abs(D) div G};
ratio(_, _) ->
    fail(invalid_rational).
decimal(C, E) when is_integer(C), is_integer(E) -> {ls_decimal, C, E}.
gcd(A, 0) -> A;
gcd(A, B) -> gcd(B, A rem B).
pow(_, 0) ->
    1;
pow(A, N) when N > 0 ->
    H = pow(A, N div 2),
    case N rem 2 of
        0 -> H * H;
        1 -> H * H * A
    end.
exact(N) when is_integer(N) -> {N, 1};
exact({ls_ratio, N, D}) ->
    {N, D};
exact({ls_decimal, 0, E}) when is_integer(E) -> {0, 1};
exact({ls_decimal, C, E}) when E >= 0 -> {C * pow(10, E), 1};
exact({ls_decimal, C, E}) ->
    normalized(C, pow(10, -E));
exact({ls_float, _, _} = F) ->
    case decode_float(F) of
        {finite, Sign, Mantissa, Exponent} ->
            Signed =
                case Sign of
                    0 -> Mantissa;
                    1 -> -Mantissa
                end,
            case Exponent >= 0 of
                true -> {Signed bsl Exponent, 1};
                false -> normalized(Signed, 1 bsl (-Exponent))
            end;
        _ ->
            fail(non_finite_exact_conversion)
    end;
exact(_) ->
    fail(exact_number_required).
normalized(N, D) ->
    {ls_ratio, RN, RD} = ratio(N, D),
    {RN, RD}.
finite_decimal(N, D) ->
    {RN, RD} = normalized(N, D),
    {D2, Twos} = factors(RD, 2, 0),
    {D5, Fives} = factors(D2, 5, 0),
    require(D5 =:= 1, non_terminating_decimal),
    Scale = max(Twos, Fives),
    decimal(RN * pow(2, Scale - Twos) * pow(5, Scale - Fives), -Scale).
factors(N, Factor, Count) ->
    case N rem Factor of
        0 -> factors(N div Factor, Factor, Count + 1);
        _ -> {N, Count}
    end.

%% @doc IEEE encoding rounds an exact ratio once, to nearest with ties to
%% even, including the subnormal/normal and finite/infinity boundaries.
%% ref:ieee-754
float_bits(W, Bits) when
    (W =:= 32 orelse W =:= 64),
    is_integer(Bits),
    Bits >= 0,
    Bits < (1 bsl W)
->
    {ls_float, W, Bits};
float_bits(_, _) ->
    fail(invalid_ieee_bits).
format(32) -> {23, 8, 127};
format(64) -> {52, 11, 1023}.
float_type(32) -> <<"Float32">>;
float_type(64) -> <<"Float64">>.
decode_float({ls_float, W, Bits}) ->
    {P, E, Bias} = format(W),
    Sign = Bits bsr (W - 1),
    Exp = (Bits bsr P) band ((1 bsl E) - 1),
    Fraction = Bits band ((1 bsl P) - 1),
    case {Exp, Fraction} of
        {Top, 0} when Top =:= (1 bsl E) - 1 -> {infinity, Sign};
        {Top, _} when Top =:= (1 bsl E) - 1 -> nan;
        {0, F} -> {finite, Sign, F, 1 - Bias - P};
        {X, F} -> {finite, Sign, (1 bsl P) + F, X - Bias - P}
    end.
float_class(F) ->
    case decode_float(F) of
        nan -> nan;
        {infinity, _} -> infinity;
        {finite, _, 0, _} -> zero;
        _ -> finite
    end.
float_sign({ls_float, W, B}) -> B bsr (W - 1).
float_zero(W, S) -> float_bits(W, S bsl (W - 1)).
float_infinity(W, S) ->
    {P, E, _} = format(W),
    float_bits(W, (S bsl (W - 1)) bor (((1 bsl E) - 1) bsl P)).
float_nan(W) ->
    {P, _, _} = format(W),
    {ls_float, W, Inf} = float_infinity(W, 0),
    float_bits(W, Inf bor (1 bsl (P - 1))).
float_from_ratio(W, 0, D, ZeroSign) when D =/= 0 -> float_zero(W, ZeroSign);
float_from_ratio(W, N, D, _) when D =/= 0 ->
    {P, _, Bias} = format(W),
    S =
        case (N < 0) =/= (D < 0) of
            true -> 1;
            false -> 0
        end,
    AN = abs(N),
    AD = abs(D),
    Exp = max(1 - Bias, floor_log2(AN, AD)),
    Shift = P - Exp,
    Mantissa =
        case Shift >= 0 of
            true -> round_ratio(AN bsl Shift, AD);
            false -> round_ratio(AN, AD bsl (-Shift))
        end,
    {M, E} =
        case Mantissa >= (1 bsl (P + 1)) of
            true -> {Mantissa bsr 1, Exp + 1};
            false -> {Mantissa, Exp}
        end,
    case E > Bias of
        true ->
            float_infinity(W, S);
        false ->
            EncodedExp =
                case M < (1 bsl P) of
                    true -> 0;
                    false -> E + Bias
                end,
            float_bits(W, (S bsl (W - 1)) bor (EncodedExp bsl P) bor (M band ((1 bsl P) - 1)))
    end;
float_from_ratio(_, _, _, _) ->
    fail(zero_denominator).
floor_log2(N, D) ->
    Guess = bit_length(N) - bit_length(D),
    Below =
        case Guess >= 0 of
            true -> N < (D bsl Guess);
            false -> (N bsl (-Guess)) < D
        end,
    case Below of
        true -> Guess - 1;
        false -> Guess
    end.
bit_length(0) ->
    0;
bit_length(N) ->
    B = binary:encode_unsigned(N),
    <<Head, _/binary>> = B,
    8 * (byte_size(B) - 1) + byte_bits(Head).
byte_bits(0) -> 0;
byte_bits(N) -> 1 + byte_bits(N bsr 1).
round_ratio(N, D) ->
    Q = N div D,
    R = N rem D,
    case 2 * R > D orelse (2 * R =:= D andalso Q rem 2 =:= 1) of
        true -> Q + 1;
        false -> Q
    end.
float_from_native(W, F) when is_float(F) ->
    <<B:64/unsigned>> = <<F:64/float>>,
    to_float(float_bits(64, B), W).
float_to_native({ls_float, W, B} = F) ->
    require(
        float_class(F) =/= nan andalso float_class(F) =/= infinity,
        non_finite_native_float
    ),
    <<Native:W/float>> = <<B:W/unsigned>>,
    Native.
to_float({ls_float, W, _} = F, W) ->
    F;
to_float({ls_float, _, _} = F, W) ->
    case decode_float(F) of
        nan ->
            float_nan(W);
        {infinity, Sign} ->
            float_infinity(W, Sign);
        _ ->
            {N, D} = exact(F),
            float_from_ratio(W, N, D, float_sign(F))
    end;
to_float(V, W) ->
    {N, D} = exact(V),
    float_from_ratio(W, N, D, 0).
same_ratio({A, B}, {C, D}) -> A * D =:= C * B.
to_complex({ls_complex, _, R, I}, W) ->
    {ls_complex, W, to_float(R, W), to_float(I, W)};
to_complex(V, W) ->
    {ls_complex, W, to_float(V, W), float_zero(W, 0)}.

%% @doc Arithmetic handles IEEE exceptional values before exact finite
%% computation; it never relies on badarith to guess an infinity's sign.
%% ref:ieee-754
float_binary(Op, {ls_float, WA, _} = A, {ls_float, WB, _} = B) ->
    W = max(WA, WB),
    AF = to_float(A, W),
    BF = to_float(B, W),
    float_op(Op, W, AF, BF, float_class(AF), float_class(BF)).
float_op(_, W, _, _, nan, _) ->
    float_nan(W);
float_op(_, W, _, _, _, nan) ->
    float_nan(W);
float_op(<<"-">>, W, A, B, CA, CB) ->
    float_op(<<"+">>, W, A, negate(B), CA, CB);
float_op(<<"+">>, W, A, B, infinity, infinity) ->
    case float_sign(A) =:= float_sign(B) of
        true -> A;
        false -> float_nan(W)
    end;
float_op(<<"+">>, _, A, _, infinity, _) ->
    A;
float_op(<<"+">>, _, _, B, _, infinity) ->
    B;
float_op(<<"*">>, W, _, _, infinity, zero) ->
    float_nan(W);
float_op(<<"*">>, W, _, _, zero, infinity) ->
    float_nan(W);
float_op(<<"*">>, W, A, B, CA, CB) when CA =:= infinity; CB =:= infinity ->
    float_infinity(W, float_sign(A) bxor float_sign(B));
float_op(<<"/">>, W, _, _, infinity, infinity) ->
    float_nan(W);
float_op(<<"/">>, W, _, _, zero, zero) ->
    float_nan(W);
float_op(<<"/">>, W, A, B, infinity, _) ->
    float_infinity(W, float_sign(A) bxor float_sign(B));
float_op(<<"/">>, W, A, B, _, zero) ->
    float_infinity(W, float_sign(A) bxor float_sign(B));
float_op(<<"/">>, W, A, B, _, infinity) ->
    float_zero(W, float_sign(A) bxor float_sign(B));
float_op(Op, W, A, B, _, _) ->
    {AN, AD} = exact(A),
    {BN, BD} = exact(B),
    {N, D, ZeroSign} =
        case Op of
            <<"+">> -> {AN * BD + BN * AD, AD * BD, float_sign(A) band float_sign(B)};
            <<"*">> -> {AN * BN, AD * BD, float_sign(A) bxor float_sign(B)};
            <<"/">> -> {AN * BD, AD * BN, float_sign(A) bxor float_sign(B)};
            _ -> fail({unknown_float_operation, Op})
        end,
    float_from_ratio(W, N, D, ZeroSign).

float_compare(A, B) ->
    case {decode_float(A), decode_float(B)} of
        {nan, _} -> unordered;
        {_, nan} -> unordered;
        {{infinity, S}, {infinity, S}} -> 0;
        {{infinity, 1}, _} -> -1;
        {_, {infinity, 1}} -> 1;
        {{infinity, 0}, _} -> 1;
        {_, {infinity, 0}} -> -1;
        _ -> compare_ratios(exact(A), exact(B))
    end.
compare_ratios({A, B}, {C, D}) -> order(A * D, C * B).
order(A, B) when A < B -> -1;
order(A, B) when A > B -> 1;
order(_, _) -> 0.

negate(N) when is_integer(N) -> -N;
negate({ls_ratio, N, D}) -> ratio(-N, D);
negate({ls_decimal, C, E}) -> decimal(-C, E);
negate({ls_float, W, B}) -> float_bits(W, B bxor (1 bsl (W - 1)));
negate({ls_complex, W, R, I}) -> {ls_complex, W, negate(R), negate(I)}.

%% Generated code names each operation; TA and TB are the operands' types.
add(A, B, TA, TB) -> binary(<<"+">>, A, B, TA, TB).
subtract(A, B, TA, TB) -> binary(<<"-">>, A, B, TA, TB).
multiply(A, B, TA, TB) -> binary(<<"*">>, A, B, TA, TB).
divide(A, B, TA, TB) -> binary(<<"/">>, A, B, TA, TB).
quotient(A, B, TA, TB) -> binary(<<"quot">>, A, B, TA, TB).
remainder(A, B, TA, TB) -> binary(<<"rem">>, A, B, TA, TB).
power(A, B, TA, TB) -> binary(<<"pow">>, A, B, TA, TB).
less(A, B, TA, TB) -> binary(<<"<">>, A, B, TA, TB).
less_or_equal(A, B, TA, TB) -> binary(<<"<=">>, A, B, TA, TB).
greater(A, B, TA, TB) -> binary(<<">">>, A, B, TA, TB).
greater_or_equal(A, B, TA, TB) -> binary(<<">=">>, A, B, TA, TB).
not_equal(A, B) -> not equal(A, B).

binary(<<"==">>, A, B, _, _) ->
    equal(A, B);
binary(<<"!=">>, A, B, _, _) ->
    not equal(A, B);
binary(Op, A, B, TA, TB) when
    Op =:= <<"<">>;
    Op =:= <<"<=">>;
    Op =:= <<">">>;
    Op =:= <<">=">>
->
    Order =
        case inexact_width(TA) > 0 orelse inexact_width(TB) > 0 of
            true -> float_compare(A, B);
            false -> compare(A, B)
        end,
    relation(Op, Order);
binary(Op, A, B, TA, TB) ->
    require(
        (inexact_width(TA) > 0) =:= (inexact_width(TB) > 0),
        exact_inexact_mixing
    ),
    case max(inexact_width(TA), inexact_width(TB)) of
        0 ->
            exact_binary(Op, A, B, TA, TB);
        Width ->
            case is_complex_type(TA) orelse is_complex_type(TB) of
                true -> complex_binary(Op, to_complex(A, Width), to_complex(B, Width));
                false -> float_binary(Op, A, B)
            end
    end.
inexact_width(<<"Float32">>) -> 32;
inexact_width(<<"Float64">>) -> 64;
inexact_width(<<"Complex64">>) -> 32;
inexact_width(<<"Complex128">>) -> 64;
inexact_width(_) -> 0.
is_complex_type(<<"Complex64">>) -> true;
is_complex_type(<<"Complex128">>) -> true;
is_complex_type(_) -> false.
relation(_, unordered) -> false;
relation(<<"<">>, N) -> N < 0;
relation(<<"<=">>, N) -> N =< 0;
relation(<<">">>, N) -> N > 0;
relation(<<">=">>, N) -> N >= 0.
exact_binary(<<"pow">>, A, B, _, _) when is_integer(A), is_integer(B), B >= 0 -> pow(A, B);
exact_binary(<<"quot">>, A, B, _, _) when is_integer(A), is_integer(B), B =/= 0 -> A div B;
exact_binary(<<"rem">>, A, B, _, _) when is_integer(A), is_integer(B), B =/= 0 -> A rem B;
exact_binary(Op, A, B, TA, TB) ->
    {AN, AD} = exact(A),
    {BN, BD} = exact(B),
    {N, D} =
        case Op of
            <<"+">> -> {AN * BD + BN * AD, AD * BD};
            <<"-">> -> {AN * BD - BN * AD, AD * BD};
            <<"*">> -> {AN * BN, AD * BD};
            <<"/">> -> {AN * BD, AD * BN};
            _ -> fail({invalid_exact_operation, Op})
        end,
    R = ratio(N, D),
    case Op =:= <<"/">> orelse TA =:= <<"Rational">> orelse TB =:= <<"Rational">> of
        true ->
            R;
        false ->
            case TA =:= <<"Decimal">> orelse TB =:= <<"Decimal">> of
                true ->
                    finite_decimal(N, D);
                false ->
                    {ls_ratio, I, 1} = R,
                    I
            end
    end.
complex_binary(Op, {ls_complex, W, A, B}, {ls_complex, W, C, D}) ->
    F = fun float_binary/3,
    {R, I} =
        case Op of
            <<"+">> ->
                {F(Op, A, C), F(Op, B, D)};
            <<"-">> ->
                {F(Op, A, C), F(Op, B, D)};
            <<"*">> ->
                {F(<<"-">>, F(Op, A, C), F(Op, B, D)), F(<<"+">>, F(Op, A, D), F(Op, B, C))};
            <<"/">> ->
                Den = F(<<"+">>, F(<<"*">>, C, C), F(<<"*">>, D, D)),
                {
                    F(Op, F(<<"+">>, F(<<"*">>, A, C), F(<<"*">>, B, D)), Den),
                    F(Op, F(<<"-">>, F(<<"*">>, B, C), F(<<"*">>, A, D)), Den)
                }
        end,
    {ls_complex, W, R, I}.

%% @doc Structural equality recurses through values; native tuple equality
%% would incorrectly make a NaN equal to itself, even inside a container.
%% ref:DEC-portable-exact-arithmetic
equal({ls_float, _, _} = A, {ls_float, _, _} = B) ->
    float_compare(A, B) =:= 0;
equal({ls_complex, _, A, B}, {ls_complex, _, C, D}) ->
    equal(A, C) andalso equal(B, D);
equal({ls_complex, W, R, I}, {ls_float, _, _} = B) ->
    equal(R, B) andalso equal(I, float_zero(W, 0));
equal({ls_float, _, _} = A, {ls_complex, _, _, _} = B) ->
    equal(B, A);
equal({ls_symbol, A, _}, {ls_symbol, B, _}) ->
    A =:= B;
equal({ls_presence, T, {some, A}}, {ls_presence, T, {some, B}}) ->
    equal(A, B);
equal(
    {ls_data, <<"lawspec.collections::type::EqSet::EqSetItems">>, [A]},
    {ls_data, <<"lawspec.collections::type::EqSet::EqSetItems">>, [B]}
) ->
    same_members(A, B);
equal({ls_data, T, A}, {ls_data, T, B}) ->
    equal(A, B);
equal(A, B) when is_list(A), is_list(B) -> equal_lists(A, B);
equal(A, B) when is_tuple(A), is_tuple(B) ->
    case is_exact(A) andalso is_exact(B) of
        true -> same_ratio(exact(A), exact(B));
        false -> equal_lists(tuple_to_list(A), tuple_to_list(B))
    end;
equal(A, B) ->
    case is_exact(A) andalso is_exact(B) of
        true -> same_ratio(exact(A), exact(B));
        false -> A =:= B
    end.
same_members(Left, Right) ->
    lists:all(fun(V) -> lists:any(fun(C) -> equal(V, C) end, Right) end, Left) andalso
        lists:all(fun(V) -> lists:any(fun(C) -> equal(V, C) end, Left) end, Right).
distinct_members(Items) -> distinct_members(Items, []).
distinct_members([], _) ->
    true;
distinct_members([V | Rest], Previous) ->
    not lists:any(fun(C) -> equal(V, C) end, Previous) andalso
        distinct_members(Rest, [V | Previous]).
equal_lists([], []) -> true;
equal_lists([A | As], [B | Bs]) -> equal(A, B) andalso equal_lists(As, Bs);
equal_lists(_, _) -> false.
is_exact(N) when is_integer(N) -> true;
is_exact({ls_ratio, _, _}) -> true;
is_exact({ls_decimal, _, _}) -> true;
is_exact(_) -> false.

%% @doc Collection keys use LawSpec's order, which differs from Erlang's
%% term order. Floating values, symbols and distinct handles are not keys.
%% ref:DEC-portable-exact-arithmetic
compare(false, false) ->
    0;
compare(true, true) ->
    0;
compare(false, true) ->
    -1;
compare(true, false) ->
    1;
compare(A, B) when is_binary(A), is_binary(B) -> order(A, B);
compare({ls_raw, T, A}, {ls_raw, T, B}) ->
    compare_lists(A, B);
compare({ls_presence, T, none}, {ls_presence, T, none}) ->
    0;
compare({ls_presence, T, none}, {ls_presence, T, {some, _}}) ->
    -1;
compare({ls_presence, T, {some, _}}, {ls_presence, T, none}) ->
    1;
compare({ls_presence, T, {some, A}}, {ls_presence, T, {some, B}}) ->
    compare(A, B);
compare({ls_data, <<"Maybe::Absent">>, []}, {ls_data, <<"Maybe::Present">>, _}) ->
    -1;
compare({ls_data, <<"Maybe::Present">>, _}, {ls_data, <<"Maybe::Absent">>, []}) ->
    1;
compare({ls_data, T, A}, {ls_data, T, B}) ->
    compare_lists(A, B);
compare({ls_data, TA, _}, {ls_data, TB, _}) ->
    order(TA, TB);
compare(A, B) when is_list(A), is_list(B) -> compare_lists(A, B);
compare({ls_handle, _, _} = H, H) ->
    0;
compare(A, A) when A =:= ls_unit; A =:= ls_null; A =:= ls_undefined -> 0;
compare(A, B) ->
    require(is_exact(A) andalso is_exact(B), no_portable_order),
    compare_ratios(exact(A), exact(B)).
compare_lists([], []) ->
    0;
compare_lists([], _) ->
    -1;
compare_lists(_, []) ->
    1;
compare_lists([A | As], [B | Bs]) ->
    case compare(A, B) of
        0 -> compare_lists(As, Bs);
        Result -> Result
    end.

helper(<<"isNaN">>, [F], _, _) ->
    float_class(F) =:= nan;
helper(<<"isInfinite">>, [F], _, _) ->
    float_class(F) =:= infinity;
helper(<<"isFinite">>, [F], _, _) ->
    C = float_class(F),
    C =:= finite orelse C =:= zero;
helper(<<"isNegativeZero">>, [F], _, _) ->
    float_class(F) =:= zero andalso float_sign(F) =:= 1;
helper(<<"length">>, [V], _, _) when is_list(V) -> length(V);
helper(<<"length">>, [V], [<<"Bytes">>], _) ->
    byte_size(V);
helper(<<"length">>, [V], [<<"Text">>], _) ->
    length(unicode:characters_to_list(V));
helper(<<"length">>, [{ls_raw, _, V}], _, _) ->
    length(V);
helper(<<"real">>, [{ls_complex, _, R, _}], _, _) ->
    R;
helper(<<"imag">>, [{ls_complex, _, _, I}], _, _) ->
    I;
helper(<<"imaginary">>, [{ls_complex, _, _, I}], _, _) ->
    I;
helper(<<"conjugate">>, [{ls_complex, W, R, I}], _, _) ->
    {ls_complex, W, R, negate(I)};
helper(<<"abs">>, [V], _, _) ->
    case is_exact(V) of
        true ->
            {N, _} = exact(V),
            case N < 0 of
                true -> negate(V);
                false -> V
            end;
        false ->
            {ls_float, W, B} = V,
            float_bits(W, B band ((1 bsl (W - 1)) - 1))
    end;
helper(<<"round">>, [V, Places], _, _) when is_integer(Places) ->
    {N, D} = exact(V),
    {SN, SD} =
        case Places >= 0 of
            true -> {N * pow(10, Places), D};
            false -> {N, D * pow(10, -Places)}
        end,
    Q = round_ratio(abs(SN), SD),
    decimal(
        case SN < 0 of
            true -> -Q;
            false -> Q
        end,
        -Places
    );
helper(<<"startsWith">>, [Text, Prefix], _, _) ->
    byte_size(Text) >= byte_size(Prefix) andalso
        binary:part(Text, 0, byte_size(Prefix)) =:= Prefix;
helper(<<"endsWith">>, [Text, Suffix], _, _) ->
    byte_size(Text) >= byte_size(Suffix) andalso
        binary:part(Text, byte_size(Text) - byte_size(Suffix), byte_size(Suffix)) =:= Suffix;
helper(<<"textContains">>, [Text, Part], _, _) ->
    Part =:= <<>> orelse binary:match(Text, Part) =/= nomatch;
helper(<<"regexMatches">>, [Pattern, Text], _, _) ->
    lawspec_beam_regex:matches(Pattern, Text);
helper(<<"checked">>, [_], _, _) ->
    true;
helper(<<"select">>, [true, A, _], _, _) ->
    A;
helper(<<"select">>, [false, _, B], _, _) ->
    B;
helper(<<"compare">>, [A, B], _, _) ->
    Tag =
        case compare(A, B) of
            -1 -> <<"Less">>;
            0 -> <<"Equal">>;
            1 -> <<"Greater">>
        end,
    {ls_data, <<"lawspec.collections::type::Ordering::", Tag/binary>>, []};
helper(<<"isPresent">>, [{ls_presence, _, P}], _, _) ->
    P =/= none;
helper(<<"presentValue">>, [{ls_presence, _, {some, V}}], _, _) ->
    V;
helper(<<"presentValue">>, [{ls_presence, _, none}], _, _) ->
    fail(absent_presence_value);
helper(<<"negate">>, [V], _, _) ->
    negate(V);
%% Sequences: Text (UTF-8, indexed by code point), Bytes (indexed by octet)
%% and List. A slice is clamped to the sequence; at is a Maybe.
helper(<<"append">>, [A, B], _, _) when is_list(A), is_list(B) -> A ++ B;
helper(<<"append">>, [A, B], _, _) when is_binary(A), is_binary(B) -> <<A/binary, B/binary>>;
helper(<<"slice">>, [X, From, To], [Kind | _], _) when is_integer(From), is_integer(To) ->
    Items = sequence_items(X, Kind),
    Count = length(Items),
    I = clamp(From, 0, Count),
    J = clamp(To, I, Count),
    sequence_like(X, Kind, lists:sublist(Items, I + 1, J - I));
helper(<<"at">>, [X, I], [Kind | _], _) when is_integer(I) ->
    Items = sequence_items(X, Kind),
    case I >= 0 andalso I < length(Items) of
        true -> {ls_data, <<"Maybe::Present">>, [lists:nth(I + 1, Items)]};
        false -> {ls_data, <<"Maybe::Absent">>, []}
    end;
helper(<<"range">>, [Lo, Hi], _, _) when is_integer(Lo), is_integer(Hi) ->
    lists:seq(Lo, Hi - 1);
helper(<<"octets">>, [B], _, _) when is_binary(B) -> binary_to_list(B);
helper(<<"bytesOf">>, [Xs], _, _) when is_list(Xs) -> list_to_binary(Xs);
helper(<<"utf8">>, [T], _, _) when is_binary(T) -> T;
helper(<<"textOf">>, [N], _, _) when is_integer(N) -> integer_to_binary(N);
helper(<<"join">>, [Separator, Parts], _, _) when is_binary(Separator), is_list(Parts) ->
    iolist_to_binary(lists:join(Separator, Parts));
helper(Op, [A, B], [TA, TB], _) when Op =:= <<"quot">>; Op =:= <<"rem">>; Op =:= <<"pow">> ->
    binary(Op, A, B, TA, TB);
helper(Name, [V], _, Bits) ->
    convert(V, Name, Bits);
helper(Name, _, _, _) ->
    fail({unknown_helper, Name}).

sequence_items(X, _) when is_list(X) -> X;
sequence_items(X, <<"Text">>) when is_binary(X) -> unicode:characters_to_list(X);
sequence_items(X, _) when is_binary(X) -> binary_to_list(X).
sequence_like(X, _, Items) when is_list(X) -> Items;
sequence_like(_, <<"Text">>, Items) -> unicode:characters_to_binary(Items);
sequence_like(_, _, Items) -> list_to_binary(Items).
clamp(N, Lo, _) when N < Lo -> Lo;
clamp(N, _, Hi) when N > Hi -> Hi;
clamp(N, _, _) -> N.

new_symbol(Description) -> {ls_symbol, make_ref(), validate(Description, <<"Text">>, 64)}.
text(Units) -> unicode:characters_to_binary(checked_units(<<"Text">>, Units)).
checked_units(Kind, Units) when is_list(Units) -> [unit(Kind, U) || U <- Units].
unit(Kind, N) when is_integer(N), N >= 0 ->
    Valid =
        case Kind of
            <<"Bytes">> -> N =< 255;
            <<"CodeUnit16">> -> N =< 65535;
            <<"Utf16Text">> -> N =< 65535;
            <<"CodePoint">> -> N =< 1114111;
            <<"CodePointText">> -> N =< 1114111;
            _ -> N =< 1114111 andalso (N < 55296 orelse N > 57343)
        end,
    require(Valid, {invalid_character, Kind}),
    N;
unit(Kind, _) ->
    fail({invalid_character, Kind}).
require(true, _) -> ok;
require(false, Reason) -> fail(Reason).
fail(Reason) -> erlang:error({lawspec, Reason}).
