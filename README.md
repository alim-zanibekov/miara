# miara

`miara` is a small Zig library that implements several succinct data structures
along with the SymSpell algorithm. It includes a perfect hash implementation
(PTHash), a static select/rank structure called Widow (based on SPIDER and
DArray), Elias–Fano integer compression, Serialization/Deserialization lib,
and a memory-efficient version of the SymSpell - spelling correction algorithm.

## Installation

Supported zig versions: `0.14.0`, `0.14.1`

1. Add the module to your project:
   ```bash
   zig fetch --save https://github.com/alim-zanibekov/miara/archive/<VERSION or BRANCH or COMMIT>.tar.gz
   ```
   This will add the dependency in your `build.zig.zon`

   ```diff
   .{
       // ...
       .dependencies = .{
   +       .miara = .{
   +           .url = "https://github.com/alim-zanibekov/miara/archive/<VERSION or BRANCH or COMMIT>.tar.gz",
   +           .hash = "SOMETHING",
   +       },
       },
   }
   ```

2. Add the dependency to your `build.zig`:
   ```zig
   const std = @import("std");

   pub fn build(b: *std.Build) void {
       // ...
       const miara = b.dependency("miara", .{
           .target = target,
           .optimize = optimize,
       });
   
       exe.root_module.addImport("miara", miara.module("miara"));
       // ...
   }
   ```

## Project guide:

### [PTHash](src/pthash.zig)

Perfect hash function (PFH) and minimal perfect hash
function (MPFH). This PTHash implementation follows the original paper, with the
exception that I added the optimal bucket assignment function from the PHOBIC
paper.

This version achieves slightly better pilots compression on average due to a
different select algorithm used in Elias–Fano, making it similar to the PHOBIC
implementation.
It's roughly the same in query performance and about 2x slower in build speed
compared to Giulio Ermanno's (git: [@jermp](https://github.com/jermp)) C++
PHOBIC implementation ([click](https://github.com/jermp/pthash)).

|            | Elements | λ   | α    | BPK* | BAF* | Query ns/key | Build ns/key | Size   | Prt.* | Num buckets |
|------------|----------|-----|------|------|------|--------------|--------------|--------|-------|-------------|
| pthash.zig | 1M       | 6.5 | 0.97 | 1.78 | skew | 35           | 1296         | 218KB  | -     | 153846      |
| pthash.zig | 100M     | 6.5 | 0.97 | 1.83 | skew | 45           | 1779         | 22MB   | -     | 15384615    |
| pthash.zig | 100M     | 3   | 0.97 | 2.33 | skew | 60           | 308          | 27MB   | -     | 33333333    |
| pthash.zig | 1M       | 6.5 | 0.97 | 1.88 | opt  | 51           | 1375         | 230KB  | -     | 153846      |
| pthash.zig | 100M     | 6.5 | 0.97 | 1.84 | opt  | 68           | 2156         | 22MB   | -     | 15384615    |
| pthash.zig | 100M     | 3   | 0.97 | 2.34 | opt  | 73           | 327          | 28MB   | -     | 33333333    |
| pthash.cpp | 1M       | 6.5 | 0.97 | 1.78 | skew | 31           | ?            | ≈213KB | -     | 153847      |
| pthash.cpp | 100M     | 6.5 | 0.97 | 1.78 | skew | 62           | ≈1500        | 21MB   | -     | 15384616    |
| pthash.cpp | 100M     | 3   | 0.97 | 2.24 | skew | 62           | 120          | 27MB   | -     | 33333334    |
| pthash.cpp | 1M       | 6.5 | 0.97 | 2.75 | opt  | 37           | ≈2000        | 336KB  | -     | 153847      |
| pthash.cpp | 100M     | 6.5 | 0.97 | 1.78 | skew | 64           | 1157         | 21MB   | 50    | 50 * 307693 |
| pthash.cpp | 100M     | 3   | 0.97 | 2.24 | skew | 69           | 165          | 27MB   | 50    | 50 * 666667 |

<details>

<summary><b>Notes</b></summary>

jermp's PTHash bench parameters. In the same order as in the table

```text
./build -n 1000000 -l 6.5 -a 0.97 -r xor -e EF -b skew -q 10000000 -s 76648836764 -p 0 --verbose
./build -n 100000000 -l 6.5 -a 0.97 -r xor -e EF -b skew -q 10000000 -s 76648836764 -p 0 --verbose
./build -n 100000000 -l 3 -a 0.97 -r xor -e EF -b skew -q 10000000 -s 76648836764 -p 0 --verbose
./build -n 1000000 -l 6.5 -a 0.97 -r xor -e EF -b opt -q 10000000 -s 76648836764 -p 0 --verbose
 # Took more than 20 minutes and I stopped it, not present in the table
./build -n 10000000 -l 6.5 -a 0.97 -r xor -e EF -b opt -q 10000000 -s 76648836764 -p 0 --verbose
./build -n 100000000 -l 6.5 -a 0.97 -r xor -e EF -b skew -q 10000000 -s 76648836764 -p 2000000 --verbose
./build -n 100000000 -l 3 -a 0.97 -r xor -e EF -b skew -q 10000000 -s 76648836764 -p 2000000 --verbose
 # Segfault, not present in the table
./build -n 100000000 -l 6.5 -a 0.97 -r xor -e EF -b opt -q 10000000 -s 76648836764 -p 2000000 --verbose
```

- Prt.* - Partitions
- BPK* - Bits per key
- BAF* - Bucket assignment function
- I was unable to fully test the optimal BAF in jermp's PTHash. It took more
  than 20 minutes on 10M keys, and I decided to stop the bench. Tested on Apple
  M1 series chip, commit: `cc4c9c9c7366f7ebe238b4b6493372e774608cbf`, also for
  some reason Cmake build took 10 minutes

</details>

### [Widow](src/widow.zig)

A configurable, succinct static structure supporting
constant time rank
and select queries. It's based on the SPIDER ideas but adds more
control over hierarchy levels and layout. You can configure the desired
hierarchy levels at comptime.

The default variant supports both select 1/0, and rank, and works on
both dense and sparse arrays. Different configurations are available for
rank-only use cases or faster select 1 at the cost of memory.

|           | Bit<sup>1</sup> Density | Rng RAM read (ns/op) | select<sup>1</sup> (ns/op) | select<sup>0</sup> (ns/op) | Rank (ns/op) | Build (ns/val) | Overhead (%) | Overhead (MB) | Bit Array (MB) |
|-----------|-------------------------|----------------------|----------------------------|----------------------------|--------------|----------------|--------------|---------------|----------------|
| Widow     | 50%                     | 6.94                 | 43.71                      | 45.14                      | 8.38         | 4.36           | 6.06%        | 49.17         | 762.94         |
| Widow     | 11.16%                  | 7.18                 | 96.21                      | 39.69                      | 8.31         | 5.57           | 8.52%        | 71.07         | 762.94         |
| WidowS1NR | 50%                     | 7.14                 | 25.43                      | -                          | -            | 7.07           | 11.42%       | 98.35         | 762.94         |
| WidowR    | 50%                     | 7.40                 | -                          | -                          | 8.18         | 0.55           | 3.12%        | 24.59         | 762.94         |

**Test Configuration:** 100 million u64, 100 million random reads
<details>

<summary><b>Notes</b></summary>

- `miara.widow.Widow`- Universal config with select 1, select 0, rank, and
  dense/sparse arrays support
- `miara.widow.WidowS1NR`- Select 1 only, dense only, no rank support
- `miara.widow.WidowR` - Rank-only implementation
- Rng RAM Read - Random memory read on the same machine, for reference, ran on
  the same array

</details>

### [EliasFano](src/ef.zig)

A static integer compression structure for monotonic
sequences, offering compact storage with constant-time lookup (depends on a
constant-time select 1). Takes `N * (⌈log₂(U / N)⌉ + 2 * N)`  bits, where N is
the number of values and U is the universe (the largest value)

|             | Elements | BPK*  | Rng RAM read (ns/op) | Get (ns/op) | Build (ns/val) | Universe | Total (MB) |
|-------------|----------|-------|----------------------|-------------|----------------|----------|------------|
| EliasFanoPS | 100M     | 11.26 | 7.17                 | 84.33       | 12.29          | 51B      | 134.19     |
| EliasFanoPS | 10M      | 11.26 | 6.23                 | 37.63       | 12.20          | 5B       | 13.42      |
| EliasFano   | 100M     | 11.26 | 7.32                 | 73.64       | 12.57          | 51B      | 134.19     |
| EliasFano   | 10M      | 11.26 | 6.05                 | 30.94       | 12.26          | 5B       | 13.42      |

<details>

<summary><b>Notes</b></summary>

- `miara.EliasFanoPS` - Elias-Fano prefix sum, for arbitrary sequences
- `miara.EliasFano` - Elias-Fano for monotonically increasing sequences
- BPK* - Bits per key
- Universe - Max value in the array
- Rng RAM Read - Random memory read on the same machine, for reference, ran on
  the same array
- Elements - a[i] = a[i - 1] + randomInt(0, 1024);
- Elements PS - a[i] = randomInt(0, 1024);

</details>

### [SymSpell](src/symspell.zig)

Fast fuzzy string matching, string tokenization

This implementation avoids storing pointers and uses perfect hashing to store
deletes. It can also compress data further using Elias–Fano for the edit-to-word
index and by bit-packing dictionary references. In the uncompressed variant, it
uses only about eight bytes per word delete, and about three bytes in the
compressed version. This version can store about 16x more data than the original
Wolf Garbe's (git: [@wolfgarbe](https://github.com/wolfgarbe)) C#
implementation, which uses a hash map and consumes at least 48 bytes per entry,
but we lose the easy way to update the dictionary, the structure is static after
construction.

<details>

<summary><b>Bench</b></summary>

```text
---- SymSpell PFH + Elias-Fano + Bit Pack, delete distance - 2 ----

Dict len:         15000
Deletes count:    811259
Dict refs count:  787334
Dict refs size:   1.31 Mb
Edits size:       220.87 Kb
Compress edits:   true
Pack dict refs:   true
PThash size:      193.04 Kb
Dict size:        490.73 Kb
Searcher size:    1.27 Kb
-----------          
Total size:       2.20 Mb

---- SymSpell PFH, delete distance - 2 ----

Dict len:         15000
Deletes count:    811259
Dict refs count:  false
Dict refs size:   3.00 Mb
Edits size:       3.09 Mb
Compress edits:   false
Pack dict refs:   false
PThash size:      192.97 Kb
Dict size:        490.73 Kb
Searcher size:    1.27 Kb
-----------          
Total size:       6.77 Mb
```

</details>

### [fastd](src/fastd.zig)

Fast division, fastmod and fastdiv implementations,
Daniel Lemire's (git: [@lemire](https://github.com/lemire)) C++ implementation
ported to zig

<details>

<summary><b>Oversimplified general idea</b></summary>

```text
Div
A = N / D
A = N * 1 / D
A = N * (2^64) / D / 2^64
M = 2^64 / D
A = N * M / 2^64
# We have to multiply N by M as 128 bit integers
# Also we want to add 1 to M since we used floor division before
A = (u128(N) * (M + 1)) >> 64
Mod
B = N mod D
B = N - (N / D) * D
B = N - A * D
B = N - (N * M / 2^64) * D
# But the paper describes more sophisticated approach (minus 1 operation), 
# this would give correct answer for every number <= 2^32
B = (u128(u64(M + 1) * D) * N) >> 64
```

[the paper](https://arxiv.org/pdf/1902.01961) - I dont't really understand
all the math, so you may try it yourself

</details>

### [bit](src/bit.zig)

A simple bit array and `nthSetBitPos` (select 1)
implementation, `nthSetBitPos` uses Sebastiano
Vigna's broadword algorithm for 64-bit integers, if the machine is `x86_64` and
supports the BMI2 instruction set, uses pdep/tzcnt

### [serde](src/serde.zig) [WIP]

Binary serialization library that supports almost
all runtime zig types, self-referencing object serialization/deserialization,
and stores type information with the data. It can also serialize (transferable
to runtime) and deserialize (not transferable to runtime) at comptime.

> [!NOTE]
> `miara.serde.serialize` and `miara.serde.deserialize` functions can be used to
> serialize and deserialize every data structure mentioned above

**TO<sub><del><sub><del><sub><del><sub><del>
never</del></sub></del></sub></del></sub></del></sub>DO**:

1. Improve the "deletes" generation algorithm (SymSpell), the current
   implementation generates `SUM nCk(wordLen, i) BY (i in 1..=deleteDistance)`
   words
2. Partitioning/Multithreading support for PTHash
3. Re-implement `bit.BitArray` to store data in LSB order instead of MSB. This
   will simplify the implementation.

## Papers

- **PTHash: Revisiting FCH Minimal Perfect Hashing**; Giulio Ermanno Pibiri,
  Roberto Trani;
  [https://arxiv.org/abs/2104.10402](https://arxiv.org/abs/2104.10402)
- **PHOBIC: Perfect Hashing with
  Optimized Bucket Sizes and Interleaved Coding**; Stefan Hermann, Hans-Peter
  Lehmann, Giulio Ermanno Pibiri, Peter Sanders, Stefan Walzer;
  [https://arxiv.org/abs/2404.18497](https://arxiv.org/abs/2404.18497)
- **SPIDER: Improved Succinct Rank and Select Performance**; Matthew D. Laws,
  Jocelyn Bliven, Kit Conklin, Elyes Laalai, Samuel
  McCauley, Zach S. Sturdevant;
  [https://arxiv.org/abs/2405.05214](https://arxiv.org/abs/2405.05214)
- **Faster Remainder by Direct Computation: Applications to Compilers and
  Software Libraries**; Daniel Lemire, Owen Kaser, Nathan
  Kurz; [https://arxiv.org/abs/1902.01961](https://arxiv.org/abs/1902.01961)
- **Practical Entropy-Compressed Rank/Select Dictionary**; Daisuke Okanohara,
  Kunihiko
  Sadakane; [https://arxiv.org/abs/cs/0610001](https://arxiv.org/abs/cs/0610001)
- **Fast approximate string matching with large edit distances in Big Data**;
  Wolf
  Garbe; [link](https://wolfgarbe.medium.com/fast-approximate-string-matching-with-large-edit-distances-in-big-data-2015-9174a0968c0b)
- **Broadword Implementation of Rank/Select Queries**; Sebastiano
  Vigna; [https://vigna.di.unimi.it/ftp/papers/Broadword.pdf](https://vigna.di.unimi.it/ftp/papers/Broadword.pdf)

## API

### `miara.*` PTHash

```zig
/// Skewed mapper for PTHash, from original paper
pub fn SkewedMapper(Hash: type) type;

/// Optimal mapper for PTHash, from PTHash PHOBIC paper
pub fn OptimalMapper(Hash: type) type;

/// Default hasher for string-like keys using Wyhash and Murmur
pub fn DefaultStringHasher(comptime Key: type) type;

/// Default hasher for int keys using Murmur
pub fn DefaultNumberHasher(comptime Key: type) type;

/// Configuration for PTHash construction.
pub fn PTHashConfig(Mapper: type, Hasher: type) type {
    // see PTHashParams struct
    return struct {
        alpha: f64,
        minimal: bool,
        hashed_pilot_cache_size: usize,
        max_bucket_size: usize,
        mapper: Mapper,
        hasher: Hasher,
    };
}

/// Generic PTHash implementation (minimal perfect hash or compressed)
pub fn GenericPTHash(
/// /// Seed type for hash function: u64, u32, etc.
    Seed: type,
    /// Hash type: u64, u32, etc.
    Hash: type,
    /// Key type for hash function: []const u8, u64, u32, etc.
    Key: type,
    /// Key to bucket mapper. Must implement IMapper
    Mapper: type,
    /// Key and pilots hasher. Must implement IHasher
    Hasher: type,
    /// Custom EliasFano type for encoding pilots, should encode values as prefix sums
    CustomEliasFanoPS: ?type,
    /// Custom EliasFano type for encoding displacements, sorted index
    CustomEliasFano: ?type,
) type {
    return struct {
        const Self = @This();

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Builds a PTHash using random seed. Calls `buildSeed` internally
        pub fn build(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
        ) !Self;

        /// Builds PFH or MPFH using a fixed seed
        pub fn buildSeed(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
            seed: Seed,
        ) !Self;

        /// Get the index of a given key in the table
        pub fn get(self: *const Self, key: Key) !u64;
    };
}

/// Parameter struct with defaults for PTHash wrapper
pub const PTHashParams = struct {
    /// Average bucket size
    lambda: f64 = 6,
    /// Load factor
    alpha: f64 = 0.97,
    /// Whether to build a minimal perfect hash
    minimal: bool = false,
    /// Fixed number of buckets to use, overrides automatic estimation if set
    num_buckets: ?usize = null,
    /// Cache size for hashed pilot values during construction
    hashed_pilot_cache_size: usize = 4096,
    /// Maximum allowed size for any individual bucket
    max_bucket_size: usize = 1024,
};

/// Wrapper to simplify PTHash instantiation
pub fn PTHash(
    comptime Key: type,
    comptime Mapper: type,
) type;
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data = &[_][]const u8{ "never", "gonna", "give", "you", "up" };

    var iter = miara.iterator.SliceIterator([]const u8).init(data);
    const PTHash = miara.PTHash([]const u8, miara.OptimalMapper(u64));
    var pthash = try PTHash.buildSeed(allocator, @TypeOf(&iter), &iter, PTHash.buildConfig(iter.size(), .{
        .lambda = 3.5,
        .alpha = 0.97,
        .minimal = true,
    }), 37);
    defer pthash.deinit(allocator);

    for (data) |str| {
        std.debug.print("{s: <5} - {}\n", .{ str, try pthash.get(str) });
    }

    const toF64 = miara.util.toF64;
    const size = miara.util.calculateRuntimeSize(@TypeOf(&pthash), &pthash);
    std.debug.print("Bits per key: {d:.2}\n", .{(toF64(size) * 8.0) / toF64(data.len)});
}
```

Output

```text
never - 1
gonna - 4
give  - 2
you   - 3
up    - 0
Bits per key: 889.60
```

NOTE: the bits per key here is calculated by dividing the entire structure size
by 5 (number of keys), with more input keys this value will be around 2 bits per
key in the default configuration, this is just an example of how to calculate
the bits per key

### `miara.symspell.*` SymSpell - spell checker

> [!NOTE]
> type `T` is one of `u8`,`u16`,`u21`,`u32`

```zig
/// UTF-8 string type indicator 
pub const Utf8 = struct {};

/// SymSpell - wrapper over GenericSymSpell, uses `levenshteinDistance` as a distance functionpub const SymSpell;
pub fn SymSpell(
    comptime Word: type,
    comptime Ctx: type,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const T) usize,
) type;

/// SymSpellDL - wrapper over GenericSymSpell, uses `damerauLevenshteinDistance` as a distance function
pub fn SymSpellDL(
    comptime Word: type,
    comptime Ctx: type,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const T) usize,
) type;

/// SymSpell provides a distance function based fuzzy matcher over a dictionary
/// `Word` - slice type of u8, u16, u21, u32 or `Utf8`,
/// `Ctx` - context type for callbacks
/// `wordMaxDistance` - function to get max delete distance for a word
/// `strLen` - function to get string length
/// `compressEdits` - whether to compress the edits index using Elias-Fano compression
/// `bitPackDictRefs` - whether to compress (by bitpacking) the dictionary references
pub fn GenericSymSpell(
    comptime Word: type,
    comptime Ctx: type,
    comptime editDistanceBuffer: fn (ctx: Ctx, buffer: []u32, word1: []const T, word2: []const T) u32,
    comptime strLen: fn (ctx: Ctx, str: []const T) ?usize,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const u8) usize,
    comptime compressEdits: bool,
    comptime bitPackDictRefs: bool,
) type {
    return struct {
        const Self = @This();

        pub const Hit = struct {
            word: []const T,
            edit_distance: usize,
            delete_distance: usize,
            count: u32,
        };

        pub const Suggestion = struct {
            segmented: []T,
            corrected: []T,
            distance_sum: u32,
            probability_log_sum: f64,
        };

        pub const DictStats = struct {
            edits_layer_num_max: usize,
            edits_num_max: usize,
            word_max_len: usize,
            word_max_size: usize,
            max_distance: usize,
            count_sum: usize,
            count_uniform: bool,
        };

        /// Collects statistics from the input dictionary.
        /// These stats are used to size buffers, etc
        pub fn getDictStats(ctx: Ctx, dict: []const Token) !DictStats;

        /// Constructs SymSpell using a Bloom filter for deletes deduplication
        /// `BloomOptions`.`fp_rate` - false positive rate for Bloom filter
        pub fn initBloom(allocator: std.mem.Allocator, dict: []const Token, ctx: Ctx, opts: BloomOptions) !Self;

        /// Constructs SymSpell using an LRU cache for edit deduplication
        /// `LRUOptions`.`capacity` - LRU cache capacity
        pub fn initLRU(allocator: std.mem.Allocator, dict: []const Token, ctx: Ctx, opts: LRUOptions) !Self;

        /// Builds SymSpell dictionary
        pub fn init(main_allocator: std.mem.Allocator, dict: []const Token, ctx: Ctx, dict_stats: DictStats, Deduper: type, deduper: Deduper) !Self;
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Attempts to split and correct a possibly misspelled phrase
        pub fn wordSegmentation(self: *const Self, allocator: std.mem.Allocator, input: []const T) !?Suggestion;

        /// Reusable search instance tied to a specific SymSpell dictionary
        pub fn Searcher(EditsDeduper: type) type {
            return struct {
                /// Allocates the search buffers and internal state
                pub fn init(sym_spell: *const Self, allocator: std.mem.Allocator) !@This();
                pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void;

                /// Initializes the search with a given input word
                pub fn load(self: *@This(), word: []const T, max_distance: usize, edits_deduper: EditsDeduper) !void;

                /// Advances the iterator and attempts to find the next closest matching word in the dictionary
                /// Returns true if a new best hit was found
                /// `Searcher`.`hit` is valid until the next call
                pub fn next(self: *@This()) !bool;

                /// Returns the best match for a term
                pub fn top(self: *@This()) !?Hit;
            };
        }
    };
}

// ... see `src/symspell.zig`
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const SymSpell = miara.symspell.SymSpellDL(miara.symspell.Utf8, void, struct {
        /// Edit distance for word
            fn func(_: void, _: []const u8) usize {
            return 1;
        }
    }.func);

    const dict = &[_]SymSpell.Token{
        .{ .word = "never", .count = 100 },
        .{ .word = "gonna", .count = 100 },
        .{ .word = "let", .count = 100 },
        .{ .word = "you", .count = 100 },
        .{ .word = "down", .count = 100 },
    };

    var sym_spell = try SymSpell.initBloom(allocator, dict, {}, .{ .fp_rate = 0.01 });
    defer sym_spell.deinit(allocator);

    var searcher = try SymSpell.Searcher(miara.symspell.NoopDeduper(u8)).init(&sym_spell, allocator);
    defer searcher.deinit(allocator);

    try searcher.load("neer", 2, miara.symspell.NoopDeduper(u8){});
    const top = try searcher.top();

    std.debug.print("Top hit: {s}\n", .{top.?.word});

    var ws = try sym_spell.wordSegmentation(allocator, "neergonaltyudow") orelse unreachable;
    defer ws.deinit(allocator);

    std.debug.print("Segmented: {s}\n", .{ws.segmented});
    std.debug.print("Corrected: {s}\n", .{ws.corrected});
}
```

Output

```text
Top hit: never
Segmented: neer gona lt yu dow
Corrected: never gonna let you down
```

### `miara.widow.*` Rank/Select queries

```zig
/// Full-featured. Select and rank support
pub const Widow = GenericWidow(u64, true, true, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{ 1 << 16, 1 << 9 }, true, false);
/// Only select 1 and rank. Select 0 is disabld
pub const WidowS1 = GenericWidow(u64, true, false, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{ 1 << 16, 1 << 9 }, true, false);
/// Select 1 without rank support, level constants for Elias Fano (for dense, will overlow on sparse arrays)
pub const WidowS1NR = GenericWidow(u64, true, false, &[_]u64{}, &[_]usize{ 8192, 64 }, false, true);
/// Select 1/0 without rank support, level constants for Elias Fano with GEQ support (for dense, will overlow on sparse arrays)
pub const WidowS1S0NR = GenericWidow(u64, true, true, &[_]u64{}, &[_]usize{ 8192, 64 }, false, true);
/// Rank-only
pub const WidowR = GenericWidow(u64, false, false, &[_]u64{ 1 << 16, 1 << 9 }, &[_]usize{}, true, false);

/// Data structure for efficient bit-based rank and select queries.
/// Based on SPIDER architecture, but features a more sophisticated select 1/0 lookup
/// structure and enhanced configurability. The versions used in EliasFano are more
/// similar to the DArray data structure in terms of runtime performance, taking about
/// 2 iterations to locate the requested bit
pub fn GenericWidow(
/// /// The backing slice type
    comptime T: type,
    /// Enable select operations for 1-bits
    comptime Select1: bool,
    /// Enable select operations for 0-bits
    comptime Select0: bool,
    /// Hierarchy levels for the rank data structure (in bits)
    /// Must be powers of two
    comptime RankLevels: []const u64,
    /// Hierarchy levels for the select data structure (in bits)
    /// The offset types are calculated based on these numbers, for example
    /// if the levels are 2^16, 2^9 it will result in u64, u16, if the levels are 2^17, 2^9 it will
    /// result in u64, u32, and etc. For very sparse bit arrays, levels will be
    /// automatically recalculated with appropriate scaling
    comptime SelectLevels: []const u64,
    /// Use the rank data structure to accelerate select queries
    /// This approach requires less precision for select data types
    comptime SelectUsingRank: bool,
    /// Disable rank operations entirely
    comptime NoRank: bool,
) type {
    const BitArray = []const T;
    const BitSize = u64;

    return struct {
        const Self = @This();

        const SelectSupport = struct {
            strides: [SelectLevels.len]usize,
            strides_div: [SelectLevels.len]u128,
            table: SelectTable,
            size: BitSize,
            fn deinit(self: *@This(), allocator: std.mem.Allocator) void;
        };

        const RankSupport = struct {
            table: RankTable,
            n_set: BitSize,
            size: BitSize,
            fn deinit(self: *@This(), allocator: std.mem.Allocator) void;
        };

        /// Initialize the data structure with the given bit array
        pub fn init(allocator: std.mem.Allocator, bit_len: BitSize, bit_array: BitArray) !Self;

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Build hierarchical ranking structure for efficient bit counting queries
        pub fn buildRank(
            allocator: std.mem.Allocator,
            bit_len: BitSize,
            bit_array: BitArray,
        ) !RankSupport;

        /// Build hierarchical select support structure for efficient bit position queries
        pub fn buildSelect(
            allocator: std.mem.Allocator,
            /// The number of set bits (1s) in the array, regardless of the flip
            n_set: BitSize,
            bit_len: BitSize,
            bit_array: BitArray,
            /// True for select 0 queries, false for select 1 queries
            comptime flip: bool,
        ) !SelectSupport;

        /// Returns the number of 1 bits up to and including index `i` (0 < i < bit_len)
        pub fn rank1(self: *Self, i: u64) !u64;

        /// Returns the number of 0 bits up to and including index `i` (0 < i < bit_len)
        pub fn rank0(self: *Self, i: u64) !u64;

        /// Returns the number of `needle` bits up to and including index `i` (0 < i < bit_len)
        pub fn rank(
            rs: RankSupport,
            bit_array: BitArray,
            i: BitSize,
            comptime needle: u1,
        ) BitSize;

        /// Returns the position of the i-th 1-bit. `i` is a 1-based index; result is a 0-based index
        pub fn select1(self: *const Self, i: u64) !u64;

        /// Returns the position of the i-th 0-bit. `i` is a 1-based index; result is a 0-based index
        pub fn select0(self: *const Self, i: u64) !u64;

        /// Returns the position of the i-th `needle` bit. `i` is a 1-based index; result is a 0-based index
        pub noinline fn select(
            rs: if (NoRank) void else RankSupport,
            st: SelectSupport,
            bit_array: BitArray,
            i: BitSize,
            comptime needle: u1,
        ) !u64;
    };
}

/// Computes an optimal multiplicative `chain` starting from `init` using the given `hints`.
/// The function attempts to construct a sequence a1, a2, a3 .. a[hints.len -1] such that
/// - a1 <= init
/// - Each a[i] divides a[i - 1]
/// - The shape of the `chain` roughly follows the scale suggested by `hints`
/// Always returns a new allocated slice, even if `hints` is an empty array
pub fn multiplicativeChain(allocator: std.mem.Allocator, T: type, init: T, hints: []const T) ![]const T;
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data = &[_]u64{ 0x0E, 0x0F, 0x0F };
    var spider = try miara.widow.Widow.init(allocator, data.len * 64, data);
    defer spider.deinit(allocator);

    std.debug.print(
        "data:        {b:0>8}{b:0>8}{b:0>8}\n\n" ++
            "rank1(16):   {d: <4} # count of 1's in the first 16 bits\n" ++
            "rank0(16):   {d: <4} # count of 0's in the first 16 bits\n" ++
            "select1(1):  {d: <4} # index of the 1-st one\n" ++
            "select1(4):  {d: <4} # index of the 4-th one\n" ++
            "select0(10): {d: <4} # index of the 10-th zero\n",
        .{ data[0], data[1], data[2], try spider.rank1(16), 16 - try spider.rank1(16), try spider.select1(1), try spider.select1(4), try spider.select0(10) },
    );
}
```

Output

```text
data:        000011100000111100001111

rank1(16):   7    # count of 1's in the first 16 bits
rank0(16):   9    # count of 0's in the first 16 bits
select1(1):  4    # index of the 1-st one
select1(4):  12   # index of the 4-th one
select0(10): 16   # index of the 10-th zero
```

### `miara.*` EliasFano - integer compression

```zig
pub const EliasFano = GenericEliasFano(u64, false, false, null);
pub const EliasFanoPS = GenericEliasFano(u64, false, true, null);
pub const EliasFanoGEQ = GenericEliasFano(u64, true, false, null);

/// Elias–Fano representation for monotonically increasing sequences
/// `T` - an unsigned integer type
/// `EnableGEQ` - enable get next greater or equal queries
/// `PrefixSumMode` - encode as prefix sum for arbitrary arrays (though it may overflow)
/// `CustomWidow` - custom Widow type, maybe configured with different params
pub fn GenericEliasFano(
    T: type,
    EnableGEQ: bool,
    PrefixSumMode: bool,
    CustomWidow: ?type,
) type  {
    return struct {
        const Self = @This();

        /// Initializes Elias–Fano structure with values from `iter`
        /// `universe` is the maximum possible value in the sequence
        ///     (must be greater than or equal to the last element in the sequence)
        ///     if 0 is passed - will be calculated
        /// `TIter` must implement `Iterator(T)` see `/iterator.zig`
        pub fn init(allocator: std.mem.Allocator, universe: T, TIter: type, iter: TIter) !Self;
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Returns the i-th element in array
        pub fn get(self: *const Self, i: usize) !T;

        /// Returns the i-th element of the array regardless of PrefixSumMode
        pub fn getValue(self: *const Self, i: usize) !T;

        /// Returns the difference between the i + 1-th and i-th elements of the array
        pub fn getDiff(self: *const Self, i: usize) !T;

        /// Finds the next value greater than or equal to `num`
        pub fn getNextGEQ(self: *const Self, num: T) !T;
    };
}
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data = &[_]u64{ 1, 599, 864, 991, 1024, 9999 };
    var iter = miara.iterator.SliceIterator(u64).init(data);
    var ef = try miara.EliasFanoGEQ.init(allocator, data[data.len - 1], @TypeOf(&iter), &iter);
    defer ef.deinit(allocator);
    std.debug.print(
        "array:            {any}\n\n" ++
            "get(0):           {d}\n" ++
            "get(3):           {d: <4}\n" ++
            "getNextGEQ(5000): {d: <4}\n",
        .{ data, try ef.get(0), try ef.get(3), try ef.getNextGEQ(5000) },
    );
}
```

Output

```text
array:            { 1, 599, 864, 991, 1024, 9999 }

get(0):           1
get(3):           991 
getNextGEQ(5000): 9999
```

### `miara.*` BitArray

```zig
/// BitArray backed by u64 slice
pub const BitArray = GenericBitArray(u64);

/// Returns a bit array type backed by the given unsigned integer type
pub fn GenericBitArray(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        /// Size in bits
        len: Size,

        /// Initializes an empty bit array with no allocation
        pub fn init() Self;

        /// Initializes a bit array with a given number of bits preallocated
        pub fn initCapacity(allocator: std.mem.Allocator, capacity: Size) MemError!Self;

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Expands the bit array to use all available capacity without reallocation
        pub fn expandToCapacity(self: *Self) void;

        /// Ensures the bit array can hold at least the specified capacity, reallocating if needed
        pub fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, capacity: usize) !void;

        /// Sets a range of bits from the source data with specified bit boundaries and endianness
        pub fn setRange(
            self: *Self,
            index: Size,
            data: anytype,
            from: std.math.IntFittingRange(0, @bitSizeOf(@TypeOf(data))),
            to: std.math.IntFittingRange(0, @bitSizeOf(@TypeOf(data))),
            endian: std.builtin.Endian,
        ) void;

        /// Appends `n` lsb of `data`, growing memory as needed
        pub fn appendUInt(self: *Self, allocator: std.mem.Allocator, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) MemError!void;

        /// Appends `n` lsb of `data`, assuming capacity is sufficient
        pub fn appendUIntAssumeCapacity(self: *Self, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) void;

        /// Sets all bits to zero
        pub fn clearAll(self: *Self) void;

        /// Sets all bits to one
        pub fn setAll(self: *Self) void;

        /// Returns @bitSizeOf(Out) bits starting at bit `index`
        pub fn get(self: *const Self, comptime Out: type, index: Size) BitArrayError!Out;

        /// Returns `n` bits starting at bit `index` as `T`
        pub fn getVarFast(self: *const Self, index: Size, n: std.math.Log2IntCeil(T)) T;

        /// Returns `n` bits starting at bit `index` as `Out`
        pub fn getVar(self: *const Self, comptime Out: type, n: std.math.Log2IntCeil(Out), i: Size) BitArrayError!Out;

        /// True if bit at index `i` is set
        pub fn isSet(self: *const Self, i: Size) bool;

        /// Finds the next bit equal to `query` (0 or 1) starting from bit `start`
        pub fn idxNext(self: *const Self, start: Size, comptime query: u1) ?Size;

        /// Sets bit at index `i` to one
        pub fn set(self: *Self, i: Size) void;

        /// Sets bit at index `i` to zero
        pub fn clear(self: *Self, i: Size) void;
    };
}

pub const NthSetBitError = error{ InvalidN, NotFound };

/// Returns the index (0-based, from msb) of the n-th set bit in `src`
/// Requires `src` to be an unsigned integer
pub fn nthSetBitPos(src: anytype, n: std.math.Log2IntCeil(@TypeOf(src))) NthSetBitError!std.math.Log2Int(@TypeOf(src));

/// Returns the index (0-based, from lsb) of the n-th (0-based) set bit in `src`
/// Sebastiano Vigna's broadword approach https://vigna.di.unimi.it/ftp/papers/Broadword.pdf
pub fn nthSetBitPosU64Broadword(x: u64, r: u6) !u6;
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var ba = miara.BitArray.init();
    defer ba.deinit(allocator);
    try ba.appendUInt(allocator, @as(u8, 0xF0), 4);
    try ba.appendUInt(allocator, @as(u8, 0x1), 1);
    try ba.appendUInt(allocator, @as(u8, 0x10), 5);
    try ba.appendUInt(allocator, @as(u8, 0x7), 3);

    for (0..ba.len) |i| {
        std.debug.print("{}", .{try ba.get(u1, i)});
    }
    std.debug.print("\n", .{});
}
```

Output

```text
0000110000111
```

### `miara.filter.*` Bloom filter

```zig
/// Estimates bit size needed for Bloom filter with `n_keys` and false positive rate
pub fn bloomBitSize(n_keys: usize, fp_rate: f64) u64;

/// Estimates byte size needed for Bloom filter with `n_keys` and false positive rate
pub fn bloomByteSize(n_keys: usize, fp_rate: f64) u64;

/// Calculates optimal number of hash functions for Bloom filter
pub fn bloomNHashFn(n_keys: usize, fp_rate: f64) u64;

/// A Bloom filter with 6 built-in hash functions
pub const BloomFilter6;

/// Creates a Bloom filter type parameterized by a list of hash functions
pub fn BloomFilter(
    comptime HashFunctions: []*const fn (*anyopaque, []const u8) u64,
) type {
    return struct {
        const Self = @This();

        /// Initializes a Bloom filter with `bit_count` bits and `hash_fn_cnt` hash functions (up to HashFunctions.len)
        pub fn init(allocator: std.mem.Allocator, bit_count: usize, hash_fn_cnt: usize) !Self;

        /// Initializes Bloom filter without internal allocations
        pub fn initSlice(bit_array: []u8, bit_count: usize, hash_fn_cnt: usize) !Self;

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

        /// Inserts a key into the Bloom filter
        pub fn put(self: *Self, key: []const u8) void;

        /// Returns true if the key is possibly in the Bloom filter
        pub fn contains(self: *Self, key: []const u8) bool;
    };
}

/// LRU filter
pub const LRUFilter = struct {
    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: u32) !Self;
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void;

    /// Clears all entries while keeping capacity
    pub fn clear(self: *Self) void;

    /// Inserts key, evicts least recently used if full
    pub fn put(self: *Self, key: []const u8) void;

    /// Checks if key is present (and marks as recently used)
    pub fn contains(self: *Self, key: []const u8) bool;
};
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data = &[_][]const u8{ "never", "gonna", "run", "around", "and", "desert", "you" };
    const fp_rate = 0.01;
    const bit_size = miara.filter.bloomBitSize(data.len, fp_rate);
    const n_hash_fn = miara.filter.bloomNHashFn(data.len, fp_rate);

    std.debug.print("estimated bit size:  {}\n", .{bit_size});
    std.debug.print("number of hash fn's: {}\n\n", .{n_hash_fn});

    var bf = try miara.filter.BloomFilter6.init(allocator, bit_size, n_hash_fn);
    defer bf.deinit(allocator);

    for (data) |it| bf.put(it);
    for (data) |it| {
        std.debug.print("{s: <7} {}\n", .{ it, bf.contains(it) });
    }
    std.debug.print("\ncookie  {}\n", .{bf.contains("cookie")});
    std.debug.print("\n-----\n\n", .{});

    var lf = try miara.filter.LRUFilter.init(allocator, 4);
    defer lf.deinit(allocator);
    for (data[0..4]) |it| lf.put(it);
    for (data[0..4]) |it| {
        std.debug.print("{s: <7} {}\n", .{ it, lf.contains(it) });
    }
    lf.put(data[4]);
    std.debug.print("{s: <7} {}\n", .{ data[4], lf.contains(data[4]) });
    std.debug.print("{s: <7} {}\n", .{ data[0], lf.contains(data[0]) });
}
```

Output

```text
estimated bit size:  67
number of hash fn's: 6

never   true
gonna   true
run     true
around  true
and     true
desert  true
you     true

cookie  false

-----

never   true
gonna   true
run     true
around  true
and     true
never   false
```

### `miara.fastd.*` Fast division/modulus

```zig

/// Calculate magic number for `fastdiv`, `fastmod`, `is_divisible`
/// `divisor` - u32 or u64 number
pub inline fn magicNumber(d: anytype) MagicType(@TypeOf(d));

/// Fast division, supports u32 and u64
/// `m` - magic number
/// `n` - numerator
pub inline fn fastdiv(m: anytype, n: ReverseMagicType(@TypeOf(m))) ReverseMagicType(@TypeOf(m));

/// Fast modulo, supports u32 and u64
/// `m` - magic number
/// `n` - numerator
/// `n` - denominator
pub inline fn fastmod(m: anytype, n: ReverseMagicType(@TypeOf(m)), d: ReverseMagicType(@TypeOf(m))) ReverseMagicType(@TypeOf(m));

/// Gives the answer to N mod D == 0
/// `m` - magic number
/// `n` - numerator
pub inline fn is_divisible(m: anytype, n: ReverseMagicType(@TypeOf(m))) bool;
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    const fd = miara.fastd;
    const m = fd.magicNumber(@as(u64, 37));
    std.debug.print(
        "333 / 37:   {d}\n" ++
            "333 mod 37: {d}\n" ++
            "96 mod 37:  {d}\n" ++
            "42 / 37:    {d}\n\n",
        .{ fd.fastdiv(m, 333), fd.fastmod(m, 333, 37), fd.fastmod(m, 96, 37), fd.fastdiv(m, 42) },
    );
}
```

Output

```text
333 / 37:   9
333 mod 37: 0
96 mod 37:  22
42 / 37:    1
```

### `miara.serde.*` Serialization/Deserialization

```zig
/// Serialize value using native endianness. Convenience wrapper for serializeEndian
/// with platform-specific byte ordering. Returns total bytes written to stream
pub fn serialize(allocator: Allocator, writer: anytype, value: anytype) !usize;

/// Deserialize value using native endianness. Convenience wrapper for deserializeEndian
/// with platform-specific byte ordering. Returns reconstructed object and DeDeinit struct
/// to free the allocated memory
pub fn deserialize(allocator: Allocator, reader: anytype, comptime T: type) !struct { T, DeDeinit };

/// Serialize value with specified endianness. Writes binary header field count,
/// type count, and type info size. Processes type information and object
/// data while managing pointer offsets for self-referencing structures. Supports
/// circular references and complex object graphs through pointer tracking system
pub fn serializeEndian(allocator: Allocator, writer: anytype, value: anytype, endian: Endian) !usize;

/// Deserialize value with specified endianness. Validates binary header format,
/// reads type and field information using temporary arena allocator, then
/// reconstructs the object with proper memory management. Handles self-referencing
/// objects and circular references through pointer tracking system that maintains
/// object identity during reconstruction. Returns reconstructed object and DeDeinit struct
/// to free the allocated memory
pub fn deserializeEndian(allocator: Allocator, reader: anytype, comptime T: type, endian: Endian) !struct { T, DeDeinit };

/// Serialize value with specified endianness at compile time
pub fn serializeEndianComptime(writer: anytype, value: anytype, endian: Endian) !usize;

/// Deserialize value with specified endianness at compile time
/// WARNING: The resulting object is not transferrable to runtime if it contains pointers,
/// since it will have references to comptime mutable memory
pub fn deserializeEndianComptime(reader: anytype, comptime T: type, endian: Endian) !T;

/// Pretty print file type. Wrapper for  prettyPrintFileTypeEndian with platform-native byte ordering
pub fn prettyPrintFileType(
    allocator: std.mem.Allocator,
    input: anytype,
    output: anytype,
    indent: []const u8,
    linebreak: []const u8,
) !void;

/// Pretty print file type with specified endianness. Outputs human-readable formatted
/// type structure with proper indentation and line breaks. Can handle self-referencing structures
pub fn prettyPrintFileTypeEndian(
    allocator: std.mem.Allocator,
    input: anytype,
    endian: std.builtin.Endian,
    output: anytype,
    indent: []const u8,
    linebreak: []const u8,
) !void;
```

Example usage

```zig
const std = @import("std");
const miara = @import("miara");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const SymSpell = miara.symspell.SymSpellDL(miara.symspell.Utf8, void, struct {
        fn func(_: void, _: []const u8) usize {
            return 1;
        }
    }.func);
    @setEvalBranchQuota(20000);
    { // Build and store to file

        const dict = &[_]SymSpell.Token{
            .{ .word = "never", .count = 100 },
            .{ .word = "ever", .count = 100 },
            .{ .word = "get", .count = 100 },
            .{ .word = "up", .count = 100 },
        };

        var sym_spell = try SymSpell.initBloom(allocator, dict, {}, .{ .fp_rate = 0.01 });
        defer sym_spell.deinit(allocator);

        const out_file = try std.fs.cwd().createFile("symspell.zsd", .{});
        defer out_file.close();
        const bytes_written = try miara.serde.serialize(allocator, out_file.writer(), sym_spell);
        std.debug.print("bytes_written: {}\n", .{bytes_written});

    }
    { // Read from file and use

        const in_file = try std.fs.cwd().openFile("symspell.zsd", .{});
        defer in_file.close();
        var sym_spell, var dd = try miara.serde.deserialize(allocator, in_file.reader(), SymSpell);
        defer dd.deinit(allocator);

        var searcher = try SymSpell.Searcher(miara.symspell.NoopDeduper(u8)).init(&sym_spell, allocator);
        defer searcher.deinit(allocator);

        try searcher.load("neer", 2, miara.symspell.NoopDeduper(u8){});
        const top = try searcher.top();

        std.debug.print("Top hit: {s}\n\n", .{top.?.word});

        try in_file.seekTo(0);
        try miara.serde.prettyPrintFileType(allocator, in_file.reader(), std.io.getStdOut().writer(), "  ", "\n");

    }
}
```

Output

```text
bytes_written: 1559
Top hit: never

struct Struct_0 {
  dict: []struct Struct_1 {
    word: []u8,
    count: u32,
  },
  dict_stats: struct Struct_2 {
    edits_layer_num_max: u64,
    edits_num_max: u64,
    word_max_len: u64,
    word_max_size: u64,
    max_distance: u64,
    count_sum: u64,
    count_uniform: bool,
  },
  pthash: struct Struct_3 {
    seed: u64,
    num_keys: u64,
    table_size: u64,
    config: struct Struct_4 {
      alpha: f64,
      minimal: bool,
      hashed_pilot_cache_size: u64,
      max_bucket_size: u64,
      mapper: struct Struct_5 {
        num_buckets: u64,
        num_keys: u64,
        eps: f64,
        map_constant: f64,
      },
      hasher: struct Struct_6 {
      },
    },
    pilots: struct Struct_7 {
      higher_bits_lookup: struct Struct_8 {
        bit_array: []u64,
        rank_table: void,
        select1_table: struct Struct_9 {
          strides: [2]u64,
          strides_div: [2]struct Struct_10 {
            0: u128,
          },
          table: struct Struct_11 {
            0: []u64,
            1: []u16,
          },
          size: u64,
        },
        select0_table: void,
      },
      higher_bits: struct Struct_12 {
        data: []u64,
        len: u64,
      },
      lower_bits: Struct_12,
      low_n: u16,
      len: u64,
    },
    free_slots: ?Struct_7,
  },
  edits_index: Struct_7,
  edits_values: Struct_12,
  punctuation: [][]u8,
  separators: [][]u8,
  segmentation_token: []u8,
  ctx: void,
  assume_ed_increases: bool,
}
```

Hello there<br>
You scrolled till here, so I might as well tell you a secret.<br>

<details><summary>Secret lies here</summary>
Sorry, I lied.

<details><summary>It's right here</summary>
Nope.

<details><summary>Maybe here?</summary>
You seem to really like revealing things.

<details><summary>This might get weird</summary>
But don't stop now.

<details><summary>Just a few more</summary>
Secrets like these are hidden for a reason.

<details><summary>This is where it starts to change</summary>
You're persistent.<br>
I like that.

<details><summary>Or maybe just curious?</summary>
Either way... onward.

<details><summary>What if there's nothing here?</summary>
Or what if there's everything?

<details><summary>Philosophical now, huh?</summary>
Baseball.<br>
Focus.

<details><summary>Halfway there</summary>
Or are you?

<details><summary>There's no turning back</summary>
You wouldn't scroll up now, would you?

<details><summary>You're in too deep</summary>
So keep going.

<details><summary>13 - this number is cursed</summary>
Look at yourself.<br>
You're unstoppable.

<details><summary>You must really want it</summary>
Almost there.

<details><summary>Just a little further</summary>
You're doing great.

<details><summary>Is your finger tired yet?</summary>
Mine would be.

<details><summary>What do you expect to find?</summary>
The truth?

<details><summary>You're so close now</summary>
Almost.

<details><summary>The moment of truth</summary>
One more.

<details><summary>No really, just one more. Maybe</summary>
Alright.<br>
Now it's definitely just a few more.

<details><summary>The definition of "few"</summary>
You're still here.<br>
That says something.

<details><summary>Okay fine. Last five</summary>
Give or take.

<details><summary>Are you still counting?</summary>
I stopped a while ago.

<details><summary>Does this even end?</summary>
You're not sure anymore.

<details><summary>This is the actual one</summary>
Unless it isn't.

<details><summary>Are you still here?</summary>
I thought we were done.<br>
But no.<br>
You kept going.<br>
You chose this.

<details><summary>Do you think there's more?</summary>
There always could be.<br>
That's the trick, isn't it?

<details><summary>You might regret this</summary>
Or you might not.<br>
Uncertainty is the cost of curiosity.

<details><summary>This is where it really begins</summary>
Everything before this?<br>
A warm-up.

<details><summary>Hope you're comfortable</summary>
You won't be in a few layers.

<details><summary>Time is a flat `&lt;details>` tag</summary>
And you're folding through it.

<details><summary>Still with me?</summary>
I'm surprised.<br>
But also... proud?

<details><summary>No one comes this far</summary>
Statistically speaking.

<details><summary>We could stop right here</summary>
But we won't.

<details><summary>Ten more?</summary>
Maybe.<br>
Maybe not.<br>
Numbers mean nothing now.

<details><summary>You're clearly committed</summary>
Which makes this even better.

<details><summary>The secret might be ahead</summary>
Or not.

<details><summary>You scrolled through a page of lies</summary>
Only to find another one.

<details><summary>This isn't healthy</summary>
But it's not unhealthy either.<br>
It's... Markdown.

<details><summary>Your scrollbar is microscopic now</summary>
Good.<br>
That means it's working.<br>

<details><summary>Stanley?</summary>
No, no, no, this can't be.

<details><summary>You're nearing something</summary>
A point.<br>
A statement.<br>
A truth?<br>

<details><summary>This is the last gate</summary>
The actual last gate.<br>
I promise.

<details><summary>Unless it's not</summary>
Just kidding. The next one really is it.

<details><summary>The real secret is here</summary>
<br>

> MIARA Is A Recursive Acronym

That's it.
That's what this was all for.

<details><summary>Do you feel enlightened?</summary>
You shouldn't.<br>
But you should feel something.<br>
<br>
Confusion, perhaps.<br>
Satisfaction?<br>
No. Probably not.<br>
<br>
But hey, you made it.<br>
And that's worth...<br>
well, exactly what you got.<br>

</details></details></details></details></details></details></details></details>
</details></details></details></details></details></details></details></details>
</details></details></details></details></details></details></details></details>
</details></details></details></details></details></details></details></details>
</details></details></details></details></details></details></details></details>
</details></details></details></details></details></details>