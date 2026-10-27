%% @doc Typed text for recorded values. Basic model-domain renderings stay
%% unchanged; extended scalars have explicit forms independent of native Show,
%% inspect or string conversion. Identity labels belong to one recording.
%% ref:REQ-law-primitives ref:DEC-portable-exact-arithmetic
-module(lawspec_beam_recorded).
-export([text/2, text/3]).

text(Value, Type) -> text(Value, Type, none).
text(Value, Type, Schema) ->
    {Text, _} = render(Value, type(Type), Schema, #{}),
    iolist_to_binary(Text).

type(T) when is_binary(T) -> lawspec_beam_scalar:type(T);
type(T) -> T.

render(Vs, {<<"List">>, [T]}, S, Ids) ->
    {Parts, Next} = sequence(Vs, lists:duplicate(length(Vs), T), S, Ids),
    {["[", lists:join(", ", Parts), "]"], Next};
render({ls_presence, Kind, none}, {Kind, [_]}, _, Ids) ->
    {
        case Kind of
            <<"Nullable">> -> <<"null">>;
            <<"Optional">> -> <<"undefined">>
        end,
        Ids
    };
render({ls_presence, Kind, {some, V}}, {Kind, [T]}, S, Ids) ->
    {Part, Next} = render(V, T, S, Ids),
    Name =
        case Kind of
            <<"Nullable">> -> <<"nullable">>;
            <<"Optional">> -> <<"optional">>
        end,
    {[Name, "(", Part, ")"], Next};
render({ls_data, Tag, Fields} = V, T, S, Ids) ->
    Types = fields(V, T, S),
    {Parts, Next} = sequence(Fields, Types, S, Ids),
    Name = lists:last(binary:split(Tag, <<"::">>, [global])),
    {
        case Parts of
            [] -> Name;
            _ -> [Name, "(", lists:join(", ", Parts), ")"]
        end,
        Next
    };
render({ls_handle, Name, Identity}, {Name, []}, _, Ids) ->
    {Number, Next} = identity({handle, Name}, Identity, Ids),
    {[lists:last(binary:split(Name, <<"::">>, [global])), "#", integer_to_binary(Number)], Next};
render({ls_symbol, Identity, Description}, {<<"Symbol">>, []}, _, Ids) ->
    {Number, Next} = identity(symbol, Identity, Ids),
    {["symbol(", integer_to_binary(Number), ", ", quote(Description), ")"], Next};
render(V, {T, []}, _, Ids) ->
    {scalar(V, T), Ids}.

fields({ls_data, <<"Maybe::Absent">>, []}, {<<"Maybe">>, [_]}, _) -> [];
fields({ls_data, <<"Maybe::Present">>, [_]}, {<<"Maybe">>, [T]}, _) -> [T];
fields({ls_data, <<"Either::Left">>, [_]}, {<<"Either">>, [A, _]}, _) -> [A];
fields({ls_data, <<"Either::Right">>, [_]}, {<<"Either">>, [_, B]}, _) -> [B];
fields(V, T, S) -> lawspec_beam_schema:field_types(V, T, S).

sequence([], [], _, Ids) ->
    {[], Ids};
sequence([V | Vs], [T | Ts], S, Ids) ->
    {Part, Next} = render(V, T, S, Ids),
    {Parts, Last} = sequence(Vs, Ts, S, Next),
    {[Part | Parts], Last}.

identity(Kind, Value, Ids) ->
    Known = maps:get(Kind, Ids, #{}),
    case maps:find(Value, Known) of
        {ok, Number} ->
            {Number, Ids};
        error ->
            Number = map_size(Known) + 1,
            {Number, Ids#{Kind => Known#{Value => Number}}}
    end.

scalar(true, <<"Bool">>) ->
    <<"true">>;
scalar(false, <<"Bool">>) ->
    <<"false">>;
scalar(ls_unit, <<"Unit">>) ->
    <<"()">>;
scalar(ls_null, <<"Null">>) ->
    <<"null">>;
scalar(ls_undefined, <<"Undefined">>) ->
    <<"undefined">>;
scalar(V, <<"Text">>) ->
    quote(V);
scalar(V, <<"Char">>) ->
    quote(<<V/utf8>>);
scalar(V, <<"CodePoint">>) ->
    integer_to_binary(V);
scalar(V, <<"CodeUnit16">>) ->
    integer_to_binary(V);
scalar(V, <<"Bytes">>) ->
    units(<<"bytes">>, binary_to_list(V));
scalar({ls_raw, T, Vs}, T) ->
    units(
        case T of
            <<"CodePointText">> -> <<"codePoints">>;
            <<"Utf16Text">> -> <<"utf16">>
        end,
        Vs
    );
scalar({ls_decimal, C, E}, <<"Decimal">>) ->
    {Coefficient, Exponent} = decimal(C, E),
    [integer_to_binary(Coefficient), "e", integer_to_binary(Exponent)];
scalar({ls_ratio, N, D}, <<"Rational">>) ->
    {ls_ratio, Numerator, Denominator} = lawspec_beam_scalar:ratio(N, D),
    ["rational(", integer_to_binary(Numerator), ", ", integer_to_binary(Denominator), ")"];
scalar({ls_float, 32, _} = V, <<"Float32">>) ->
    float_text(V);
scalar({ls_float, 64, _} = V, <<"Float64">>) ->
    float_text(V);
scalar({ls_complex, 32, R, I}, <<"Complex64">>) ->
    ["Complex64(", float_text(R), ", ", float_text(I), ")"];
scalar({ls_complex, 64, R, I}, <<"Complex128">>) ->
    ["Complex128(", float_text(R), ", ", float_text(I), ")"];
scalar(V, T) when is_integer(V) ->
    case lawspec_beam_scalar:integer_type(T) of
        true -> integer_to_binary(V);
        false -> error({lawspec, {invalid_recorded_scalar, T}})
    end.

%% Equal decimal values share a recording, regardless of retained scale.
%% Divide the coefficient, never allocate a power from an unbounded exponent.
decimal(0, _) -> {0, 0};
decimal(C, E) when C rem 10 =:= 0 -> decimal(C div 10, E + 1);
decimal(C, E) -> {C, E}.

%% Native float APIs can quiet or discard NaN payloads. Record their portable
%% class, while keeping exact bits for finite values, signed zero and infinity.
float_text({ls_float, W, Bits} = V) ->
    Name = ["float", integer_to_binary(W)],
    case lawspec_beam_scalar:float_class(V) of
        nan ->
            [Name, "NaN"];
        _ ->
            Hex = string:lowercase(integer_to_binary(Bits, 16)),
            [Name, "Bits(\"", binary:copy(<<"0">>, W div 4 - byte_size(Hex)), Hex, "\")"]
    end.

units(Name, Vs) ->
    [Name, "([", lists:join(", ", [integer_to_binary(V) || V <- Vs]), "])"].
quote(B) ->
    Escaped = binary:replace(
        binary:replace(B, <<"\\">>, <<"\\\\">>, [global]), <<"\"">>, <<"\\\"">>, [global]
    ),
    [$", Escaped, $"].
