# Text Insights

A powerful command-line tool that analyzes text files and generates visual insights about word patterns, readability metrics, sentiment analysis, and semantic connections.

## Features

- Parse and analyze multiple text files or entire directories
- Generate frequency distribution charts for words, characters, and n-grams
- Calculate readability scores using various algorithms (Flesch-Kincaid, Coleman-Liau, etc.)
- Perform sentiment analysis to understand the emotional tone of documents
- Identify semantic patterns and topic clusters through n-gram analysis
- Compare multiple documents to identify similarities and differences
- Export analysis data in JSON format
- Generate visualizations including:
  - Word frequency charts
  - Character distribution charts
  - Sentiment analysis gauges
  - Word heatmaps and clouds
  - Document comparison matrices

## Installation

Make sure you have Rust installed. Then run:

```
cargo install text_insights
```

Or clone this repository and build from source:

```
git clone https://github.com/yourusername/text_insights.git
cd text_insights
cargo build --release
```

## Usage

### Basic Analysis

Analyze a single file:

```
text_insights analyze path/to/file.txt
```

Analyze a directory recursively:

```
text_insights analyze --recursive path/to/directory
```

Analyze with detailed output:

```
text_insights analyze --detailed path/to/file.txt
```

Export results to JSON:

```
text_insights analyze --output-json results.json path/to/file.txt
```

### Visualizations

Generate a word frequency chart:

```
text_insights visualize --input results.json --type word-frequency --output word_frequency.png
```

Generate a sentiment analysis visualization:

```
text_insights visualize --input results.json --type sentiment --output sentiment.png
```

Available visualization types:
- `word-frequency`: Bar chart of most common words
- `character-frequency`: Distribution of characters
- `sentence-length`: Distribution of sentence lengths
- `readability`: Chart of readability metrics
- `sentiment`: Gauge visualization of emotional tone
- `word-heatmap`: Heat map of word frequencies
- `word-cloud`: Visual representation of word frequencies
- `ngram-frequency`: Chart of common phrases

### Document Comparison

Compare multiple documents:

```
text_insights compare file1.txt file2.txt file3.txt
```

Compare with visualization:

```
text_insights compare --chart comparison.png file1.txt file2.txt
```

Export comparison results:

```
text_insights compare --output-json comparison.json file1.txt file2.txt
```

## Examples

Analyze a novel and generate a word cloud:

```
text_insights analyze --output-json moby-dick.json moby-dick.txt
text_insights visualize --input moby-dick.json --type word-cloud --output moby-dick-cloud.png
```

Compare writing styles of different authors:

```
text_insights compare --chart author-comparison.png author1/*.txt author2/*.txt
```

## Advanced Usage

Filter files by pattern:

```
text_insights analyze --recursive --pattern "*.md" path/to/directory
```

Limit the number of items in visualizations:

```
text_insights visualize --input results.json --type word-frequency --limit 30 --output chart.png
```

## Output Examples

The tool provides both console output and visual representations of the analysis:

```
=== Text Insights Analysis Summary ===
Files processed: 3
Total words: 4285
Unique words: 1342
Average words per file: 1428.33

Sentiment Analysis: Positive (0.35)

Top 10 most common words:
  1. the (203)
  2. and (143)
  3. of (138)
  4. to (91)
  5. a (88)
  6. in (62)
  7. is (54)
  8. that (42)
  9. as (36)
  10. with (32)

Readability scores:
  Reading Level: Fairly Difficult
  Flesch-Kincaid Grade: 11.24
  Flesch Reading Ease: 52.36
  Coleman-Liau Index: 10.76
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 