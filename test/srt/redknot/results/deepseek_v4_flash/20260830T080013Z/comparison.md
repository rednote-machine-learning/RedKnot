# DeepSeek V4 Flash + RedKnot output comparison

The report compares the complete generated text directly.

## 256K / hotpotqa

- Dense TTFT p50: 31.5256 s
- RedKnot TTFT p50: 15.6162 s
- TTFT speedup: 2.01877x
- Full-input major-compute saving: 0.508175
- Full-model MLA head-row saving: 0.105734
- Supplemental output target: 0 generated tokens (0 means the unchanged shortest-span benchmark)

## 256K / hotpotqa

- Dense TTFT p50: 31.4832 s
- RedKnot TTFT p50: 21.1759 s
- TTFT speedup: 1.48675x
- Full-input major-compute saving: 0.513965
- Full-model MLA head-row saving: 0.115355
- Supplemental output target: 30 generated tokens (0 means the unchanged shortest-span benchmark)

## 256K / musique

- Dense TTFT p50: 31.5139 s
- RedKnot TTFT p50: 15.1954 s
- TTFT speedup: 2.07391x
- Full-input major-compute saving: 0.520631
- Full-model MLA head-row saving: 0.0899206
- Supplemental output target: 30 generated tokens (0 means the unchanged shortest-span benchmark)

## 256K / multifieldqa_en

- Dense TTFT p50: 31.4616 s
- RedKnot TTFT p50: 21.8998 s
- TTFT speedup: 1.43661x
- Full-input major-compute saving: 0.510848
- Full-model MLA head-row saving: 0.11508
- Supplemental output target: 30 generated tokens (0 means the unchanged shortest-span benchmark)

# Ordered 256K suite outputs

The following fifteen comparisons are in the exact JSONL order: ten frozen short-answer cases, then five long-answer cases.

## Case 01: short_hotpotqa_10-row0

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Which case was brought to court first Miller v. California or Gates v. Collier ?

| Dense output | RedKnot output |
|---|---|
| <pre>[6 tokens]
Gates v. Collier</pre> | <pre>[6 tokens]
Gates v. Collier</pre> |

## Case 02: short_hotpotqa_10-row3

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Gary L. Bennett was a part of the space missions that have a primary destination of what celestial body?

| Dense output | RedKnot output |
|---|---|
| <pre>[2 tokens]
Sun</pre> | <pre>[2 tokens]
Sun</pre> |

## Case 03: short_hotpotqa_10-row6

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Thirukkalacherry is a village in which Indian town?

| Dense output | RedKnot output |
|---|---|
| <pre>[31 tokens]
Thirukkalachery is a village in Mayiladuthurai District, Tranquebar (Tharangambadi) Taluk.</pre> | <pre>[5 tokens]
Tharangambadi</pre> |

## Case 04: short_hotpotqa_10-row10

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: When was the American singer, songwriter, record producer, dancer and actress born who&#x27;s second studio album is Chapter II?

| Dense output | RedKnot output |
|---|---|
| <pre>[8 tokens]
October 13, 1980</pre> | <pre>[8 tokens]
October 13, 1980</pre> |

## Case 05: short_hotpotqa_10-row13

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Finally includes the song that peaked at what on the US Hot 100

| Dense output | RedKnot output |
|---|---|
| <pre>[3 tokens]
number five</pre> | <pre>[3 tokens]
number five</pre> |

## Case 06: short_hotpotqa_10-row16

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Which retired Argentine footballer who played as a forward was a main player for Valencia CF?

| Dense output | RedKnot output |
|---|---|
| <pre>[4 tokens]
Claudio López</pre> | <pre>[4 tokens]
Claudio López</pre> |

## Case 07: short_hotpotqa_10-row19

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Name a member of a British-American supergroup who recored a version of Nobody&#x27;s child in 1990

| Dense output | RedKnot output |
|---|---|
| <pre>[3 tokens]
George Harrison</pre> | <pre>[3 tokens]
George Harrison</pre> |

## Case 08: short_hotpotqa_10-row22

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Oak Beach, New York and Great South Bay are both situated between what same island?

| Dense output | RedKnot output |
|---|---|
| <pre>[3 tokens]
Long Island</pre> | <pre>[4 tokens]
Jones Beach Island</pre> |

## Case 09: short_hotpotqa_10-row1

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: The actor that plays Phileas Fogg in &quot;Around the World in 80 Days&quot;, co-starred with Gary Cooper in a 1939 Goldwyn Productions film based on a novel by what author?

| Dense output | RedKnot output |
|---|---|
| <pre>[5 tokens]
Charles L. Clifford</pre> | <pre>[5 tokens]
Charles L. Clifford</pre> |

## Case 10: short_hotpotqa_10-row4

- Dataset: `hotpotqa`
- Output target: `0` tokens
- Selection origin: `output_blind_standard_256k`
- Included in primary accuracy aggregate: `true`

Question: Was Atom Egoyans biggest commercial success on stage or on film?

| Dense output | RedKnot output |
|---|---|
| <pre>[2 tokens]
film</pre> | <pre>[2 tokens]
film</pre> |

## Case 11: long_hotpotqa_2-row6

- Dataset: `hotpotqa`
- Output target: `30` tokens
- Selection origin: `posthoc_long_output_showcase_20260830`
- Included in primary accuracy aggregate: `false`

Question: Thirukkalacherry is a village in which Indian town?

| Dense output | RedKnot output |
|---|---|
| <pre>[42 tokens]
Tharangambadi  
Thirukkalacherry is a small village in Mayiladuthurai District, Tranquebar (Tharangambadi) Taluk, Tamil Nadu, India.</pre> | <pre>[5 tokens]
Tharangambadi</pre> |

## Case 12: long_hotpotqa_2-row0

- Dataset: `hotpotqa`
- Output target: `30` tokens
- Selection origin: `posthoc_long_output_showcase_20260830`
- Included in primary accuracy aggregate: `false`

Question: Which case was brought to court first Miller v. California or Gates v. Collier ?

| Dense output | RedKnot output |
|---|---|
| <pre>[42 tokens]
Gates v. Collier was brought to court first.  
The documents state Gates v. Collier was a 1970–1971 case, while Miller v. California was decided in 1973.</pre> | <pre>[39 tokens]
Gates v. Collier was brought to court first.  
The documents state Gates v. Collier was filed in 1971, while Miller v. California was decided in 1973.</pre> |

## Case 13: long_musique_2-row0

- Dataset: `musique`
- Output target: `30` tokens
- Selection origin: `posthoc_long_output_showcase_20260830`
- Included in primary accuracy aggregate: `false`

Question: Who plays the wife of the producer of Here Comes the Boom in Grown Ups?

| Dense output | RedKnot output |
|---|---|
| <pre>[42 tokens]
Salma Hayek  
In Grown Ups, Salma Hayek plays Roxanne, the wife of Lenny Feder, who is a Hollywood talent agent, not the producer of Here Comes the Boom.</pre> | <pre>[21 tokens]
Salma Hayek plays the wife of the producer of Here Comes the Boom in Grown Ups.</pre> |

## Case 14: long_musique_2-row3

- Dataset: `musique`
- Output target: `30` tokens
- Selection origin: `posthoc_long_output_showcase_20260830`
- Included in primary accuracy aggregate: `false`

Question: Who is the child of the Italian navigator who sailed for England and explored the eastern coast of the continent Francisco Bojado&#x27;s birthplace is located?

| Dense output | RedKnot output |
|---|---|
| <pre>[46 tokens]
Sebastian Cabot  
Sebastian Cabot was the son of John Cabot, the Italian navigator who sailed for England. Francisco Bojado was born in Mexico, not on a continent&#x27;s eastern coast.</pre> | <pre>[44 tokens]
Sebastian Cabot  
Sebastian Cabot was the son of John Cabot, the Italian navigator who sailed for England. Francisco Bojado was born in Mexico, which is in North America.</pre> |

## Case 15: long_multifieldqa_en_1-row0

- Dataset: `multifieldqa_en`
- Output target: `30` tokens
- Selection origin: `posthoc_long_output_showcase_20260830`
- Included in primary accuracy aggregate: `false`

Question: What is the name of the most active fan club?

| Dense output | RedKnot output |
|---|---|
| <pre>[37 tokens]
South West Ultras fan club.  
The club is a de facto representative of the Malatia-Sebastia District, and its most active fan group is the South West Ultras.</pre> | <pre>[40 tokens]
South West Ultras fan club.  
The documents state that the most active group of fans is the South West Ultras fan club, composed mainly of residents from the Malatia-Sebastia District.</pre> |

