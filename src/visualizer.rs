use anyhow::{Context, Result};
use plotters::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use std::collections::HashSet;
use plotters::coord::Shift;

use crate::analyzer::AnalysisResults;

pub fn generate_visualization(
    results: &AnalysisResults,
    viz_type: &str,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    match viz_type {
        "word-frequency" => generate_word_frequency_chart(results, output_path, limit),
        "character-frequency" => generate_character_frequency_chart(results, output_path, limit),
        "sentence-length" => generate_sentence_length_chart(results, output_path),
        "readability" => generate_readability_chart(results, output_path),
        "sentiment" => generate_sentiment_chart(results, output_path),
        "word-heatmap" => generate_word_heatmap(results, output_path, limit),
        "word-cloud" => generate_word_cloud(results, output_path, limit),
        "ngram-frequency" => generate_ngram_chart(results, output_path, limit),
        _ => Err(anyhow::anyhow!("Unsupported visualization type: {}", viz_type)),
    }
}

fn generate_word_frequency_chart(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Filter out common stopwords
    let stopwords = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
    ];
    
    let filtered_words: Vec<_> = results.top_words
        .iter()
        .filter(|(word, _)| !stopwords.contains(&word.as_str()))
        .take(limit)
        .collect();
    
    if filtered_words.is_empty() {
        return Err(anyhow::anyhow!("No words to visualize after filtering"));
    }
    
    let max_count = filtered_words.first().map(|(_, count)| *count).unwrap_or(0);
    let max_count_i32 = max_count as i32;
    
    // Use a very large canvas for maximum readability
    let root = BitMapBackend::new(output_path, (1600, 900)).into_drawing_area();
    
    // Pure white background for maximum contrast
    root.fill(&WHITE)?;
    
    // Generous margins
    let chart_area = root.margin(60, 60, 60, 60);
    
    // Display fewer words for better readability
    let display_limit = filtered_words.len().min(15);
    let words_to_display = &filtered_words[0..display_limit];
    
    // Create a titled chart with bold, clear styling
    let mut chart = ChartBuilder::on(&chart_area)
        .caption(
            "WORD FREQUENCY", 
            ("sans-serif", 50).into_font().color(&RGBColor(30, 30, 85))
        )
        .set_label_area_size(LabelAreaPosition::Left, 100)  // Larger label area
        .set_label_area_size(LabelAreaPosition::Bottom, 100) // Larger label area
        .margin(15)
        .build_cartesian_2d(
            0..display_limit as i32,
            0..max_count_i32 + (max_count_i32 / 5), // More space above bars
        )?;
    
    // Configure grid and labels with very clear styling
    chart.configure_mesh()
        .disable_x_mesh()
        .light_line_style(RGBColor(240, 240, 245))
        .bold_line_style(RGBColor(220, 220, 230))
        .axis_style(ShapeStyle::from(&RGBColor(100, 100, 100)).stroke_width(2))
        .y_desc("FREQUENCY")
        .y_label_style(("sans-serif", 26).into_font().color(&RGBColor(60, 60, 80)))
        .x_label_formatter(&|x| {
            if *x < display_limit as i32 {
                // Convert words to uppercase for better readability
                words_to_display[*x as usize].0.to_uppercase()
            } else {
                String::new()
            }
        })
        .x_labels(display_limit) // Show all labels
        .label_style(("sans-serif", 20).into_font().color(&RGBColor(60, 60, 80)))
        .y_label_style(("sans-serif", 22).into_font().color(&RGBColor(60, 60, 80)))
        .x_label_style(("sans-serif", 22).into_font().color(&RGBColor(30, 30, 80)))
        .y_label_offset(15) // More space for y-axis labels
        .draw()?;
    
    // Use a clear, high-contrast color for bars
    let primary_color = RGBColor(42, 100, 246); // Strong blue
    
    // Draw horizontal gridlines for easier reading
    for y in (0..=max_count_i32).step_by((max_count_i32 / 10).max(1) as usize) {
        if y > 0 {
            chart.draw_series(std::iter::once(
                PathElement::new(
                    vec![(0, y), (display_limit as i32, y)],
                    RGBColor(235, 235, 245).stroke_width(1)
                )
            ))?;
        }
    }
    
    // Draw wide, clear bars with enough spacing
    for (i, (word, count)) in words_to_display.iter().enumerate() {
        let i_i32 = i as i32;
        let count_i32 = *count as i32;
        
        // Calculate bar width with proper spacing
        let bar_width = 0.7; // Width of each bar (out of 1.0 unit)
        let x0 = (i_i32 as f64 + (1.0 - bar_width) / 2.0) as i32;
        let x1 = (i_i32 as f64 + (1.0 - bar_width) / 2.0 + bar_width) as i32;
        
        // Draw a solid, clear bar
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, 0), (x1, count_i32)], 
                primary_color.mix(0.9).filled()
            )
        ))?;
        
        // Add a bold border
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, 0), (x1, count_i32)], 
                primary_color.stroke_width(2)
            )
        ))?;
        
        // Add large, clear count number at the top of each bar
        chart.draw_series(std::iter::once(
            Text::new(
                count.to_string(),
                (i_i32, count_i32 + max_count_i32 / 30),
                ("sans-serif", 22).into_font().color(&RGBColor(50, 50, 100))
            )
        ))?;
    }
    
    // Add a clear source label at the top
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        chart_area.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (800, 30),
            ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100))
        ))?;
    }
    
    // Add a clear summary in large text at the bottom
    let footer_text = format!(
        "Analysis of {} unique words from {} total words", 
        results.unique_words, 
        results.total_words
    );
    
    chart_area.draw(&Text::new(
        footer_text,
        (800, chart_area.dim_in_pixel().1 as i32 - 30),
        ("sans-serif", 24).into_font().color(&RGBColor(80, 80, 100))
    ))?;
    
    // Add generation date in a subtle footer
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    chart_area.draw(&Text::new(
        format!("Generated: {}", current_date),
        (1400, chart_area.dim_in_pixel().1 as i32 - 30),
        ("sans-serif", 16).into_font().color(&RGBColor(150, 150, 170))
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_character_frequency_chart(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Sort characters by frequency
    let mut char_freqs: Vec<_> = results.character_frequencies
        .iter()
        .collect();
    
    char_freqs.sort_by(|a, b| b.1.cmp(a.1));
    
    let chars_to_display = char_freqs.iter()
        .filter(|(c, _)| c.is_alphabetic()) // Only show alphabetic characters
        .take(limit)
        .collect::<Vec<_>>();
    
    if chars_to_display.is_empty() {
        return Err(anyhow::anyhow!("No characters to visualize"));
    }
    
    let max_count = chars_to_display.first().map(|(_, count)| **count).unwrap_or(0);
    
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Character Frequency", ("sans-serif", 40))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0..chars_to_display.len(),
            0..(max_count + max_count / 10),
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .y_desc("Frequency")
        .x_label_formatter(&|x| {
            if *x < chars_to_display.len() {
                chars_to_display[*x].0.to_string()
            } else {
                String::new()
            }
        })
        .x_labels(chars_to_display.len())
        .label_style(("sans-serif", 12))
        .draw()?;
    
    chart.draw_series(
        chars_to_display.iter().enumerate().map(|(i, (_, count))| {
            let color = Palette99::pick(i).mix(0.9);
            Rectangle::new([(i, 0), (i + 1, **count)], color.filled())
        })
    )?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_sentence_length_chart(
    results: &AnalysisResults,
    output_path: &Path,
) -> Result<()> {
    if results.sentence_lengths.is_empty() {
        return Err(anyhow::anyhow!("No sentence length data available"));
    }
    
    // Group sentence lengths into bins
    let max_length = *results.sentence_lengths.iter().max().unwrap_or(&0);
    let bin_size = std::cmp::max(1, max_length / 20); // Aim for around 20 bins
    
    let mut bins = vec![0; (max_length / bin_size) + 1];
    
    for &length in &results.sentence_lengths {
        let bin = length / bin_size;
        bins[bin] += 1;
    }
    
    let max_count = *bins.iter().max().unwrap_or(&0);
    
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Sentence Length Distribution", ("sans-serif", 40))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0..bins.len(),
            0..(max_count + max_count / 10),
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .y_desc("Number of Sentences")
        .x_desc("Sentence Length (words)")
        .x_label_formatter(&|x| format!("{}-{}", x * bin_size, (x + 1) * bin_size - 1))
        .x_labels(bins.len().min(20)) // Limit number of labels
        .label_style(("sans-serif", 12))
        .draw()?;
    
    chart.draw_series(
        bins.iter().enumerate().map(|(i, &count)| {
            let color = HSLColor(0.3, 0.7, 0.5);
            Rectangle::new([(i, 0), (i + 1, count)], color.filled())
        })
    )?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_readability_chart(
    results: &AnalysisResults,
    output_path: &Path,
) -> Result<()> {
    if results.readability_scores.is_empty() {
        return Err(anyhow::anyhow!("No readability data available"));
    }
    
    // Convert the HashMap to a Vec for easier manipulation
    let scores: Vec<_> = results.readability_scores
        .iter()
        .filter(|(name, _)| *name != "Words per Sentence") // Exclude some metrics
        .collect();
    
    if scores.is_empty() {
        return Err(anyhow::anyhow!("No readability metrics to display"));
    }
    
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find min and max for better scaling
    let min_score = scores.iter().map(|(_, &v)| v).fold(f64::INFINITY, f64::min);
    let max_score = scores.iter().map(|(_, &v)| v).fold(f64::NEG_INFINITY, f64::max);
    
    let padding = (max_score - min_score).abs() * 0.1;
    let y_range = (min_score - padding)..(max_score + padding);
    
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Readability Scores", ("sans-serif", 40))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 120)
        .build_cartesian_2d(
            0..scores.len(),
            y_range,
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .y_desc("Score")
        .x_label_formatter(&|x| {
            if *x < scores.len() {
                scores[*x].0.clone()
            } else {
                String::new()
            }
        })
        .x_labels(scores.len())
        .label_style(("sans-serif", 12))
        .draw()?;
    
    chart.draw_series(
        scores.iter().enumerate().map(|(i, (_name, &score))| {
            EmptyElement::at((i, score))
                + Circle::new((0, 0), 5, ShapeStyle::from(&BLUE).filled())
                + Text::new(format!("{:.2}", score), (0, -15), ("sans-serif", 12).into_font())
        })
    )?;
    
    chart.draw_series(LineSeries::new(
        scores.iter().enumerate().map(|(i, (_, &score))| (i, score)),
        &BLUE,
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_sentiment_chart(
    results: &AnalysisResults,
    output_path: &Path,
) -> Result<()> {
    // Check if sentiment data is available
    if !results.sentiment_scores.contains_key("Overall Sentiment") {
        return Err(anyhow::anyhow!("No sentiment data available"));
    }
    
    let sentiment = *results.sentiment_scores.get("Overall Sentiment").unwrap();
    
    let root = BitMapBackend::new(output_path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Set up the gauge chart
    let _gauge_range = -1.0..1.0; // Unused but kept for clarity
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Sentiment Analysis", ("sans-serif", 40))
        .build_cartesian_2d(-1.2..1.2, 0.0..1.0)?;
    
    chart.configure_mesh().disable_mesh().draw()?;
    
    // Draw gauge background
    let gauge_height = 0.3;
    let gauge_y = 0.5;
    
    // Draw negative side (red gradient)
    for x in -100..0 {
        let x_pos = x as f64 / 100.0;
        let color_intensity = (-x_pos).powf(0.5); // Square root for more even gradient
        
        let color = RGBColor(
            (200.0 + 55.0 * color_intensity) as u8,
            (100.0 * (1.0 - color_intensity)) as u8,
            (100.0 * (1.0 - color_intensity)) as u8,
        );
        
        chart.draw_series(std::iter::once(
            Rectangle::new([(x_pos, gauge_y - gauge_height / 2.0), (x_pos + 0.01, gauge_y + gauge_height / 2.0)],
            color.filled())
        ))?;
    }
    
    // Draw positive side (green gradient)
    for x in 0..100 {
        let x_pos = x as f64 / 100.0;
        let color_intensity = x_pos.powf(0.5); // Square root for more even gradient
        
        let color = RGBColor(
            (100.0 * (1.0 - color_intensity)) as u8,
            (200.0 + 55.0 * color_intensity) as u8,
            (100.0 * (1.0 - color_intensity)) as u8,
        );
        
        chart.draw_series(std::iter::once(
            Rectangle::new([(x_pos, gauge_y - gauge_height / 2.0), (x_pos + 0.01, gauge_y + gauge_height / 2.0)],
            color.filled())
        ))?;
    }
    
    // Draw center line
    chart.draw_series(std::iter::once(
        PathElement::new(vec![(0.0, gauge_y - gauge_height / 2.0), (0.0, gauge_y + gauge_height / 2.0)],
        BLACK.stroke_width(2))
    ))?;
    
    // Draw border
    chart.draw_series(std::iter::once(
        Rectangle::new([(-1.0, gauge_y - gauge_height / 2.0), (1.0, gauge_y + gauge_height / 2.0)],
        BLACK.stroke_width(2))
    ))?;
    
    // Draw labels
    chart.draw_series(std::iter::once(
        Text::new("Very Negative", (-0.95, gauge_y + gauge_height + 0.1), ("sans-serif", 20))
    ))?;
    
    chart.draw_series(std::iter::once(
        Text::new("Neutral", (0.0, gauge_y + gauge_height + 0.1), ("sans-serif", 20))
    ))?;
    
    chart.draw_series(std::iter::once(
        Text::new("Very Positive", (0.7, gauge_y + gauge_height + 0.1), ("sans-serif", 20))
    ))?;
    
    // Draw pointer
    let clamped_sentiment = sentiment.max(-1.0).min(1.0);
    let triangle_size = 0.1;
    
    chart.draw_series(std::iter::once(
        PathElement::new(
            vec![
                (clamped_sentiment, gauge_y - gauge_height / 2.0 - triangle_size),
                (clamped_sentiment - triangle_size / 2.0, gauge_y - gauge_height / 2.0 - triangle_size * 2.0),
                (clamped_sentiment + triangle_size / 2.0, gauge_y - gauge_height / 2.0 - triangle_size * 2.0),
            ],
            BLACK.filled(),
        )
    ))?;
    
    // Draw sentiment value
    chart.draw_series(std::iter::once(
        Text::new(
            format!("{:.2}", sentiment),
            (clamped_sentiment, gauge_y - gauge_height / 2.0 - triangle_size * 3.0),
            ("sans-serif", 25).into_font(),
        )
    ))?;
    
    // Draw sentiment description
    let (description, color) = match sentiment {
        s if s >= 0.6 => ("Very Positive", GREEN.to_rgba()),
        s if s >= 0.2 => ("Positive", RGBColor(100, 200, 100).to_rgba()),
        s if s > -0.2 => ("Neutral", RGBColor(100, 100, 100).to_rgba()),
        s if s > -0.6 => ("Negative", RGBColor(200, 100, 100).to_rgba()),
        _ => ("Very Negative", RED.to_rgba()),
    };
    
    chart.draw_series(std::iter::once(
        Text::new(
            description,
            (clamped_sentiment, gauge_y - gauge_height / 2.0 - triangle_size * 5.0),
            ("sans-serif", 30).into_font().color(&color),
        )
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_word_heatmap(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Filter out common stopwords and keep top N words
    let stopwords = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
    ];
    
    let filtered_words: Vec<_> = results.top_words
        .iter()
        .filter(|(word, _)| !stopwords.contains(&word.as_str()))
        .take(limit.min(20)) // Maximum 20 words for readability
        .collect();
    
    if filtered_words.is_empty() {
        return Err(anyhow::anyhow!("No words to visualize after filtering"));
    }
    
    let max_count = filtered_words.iter().map(|(_, count)| *count).max().unwrap_or(1);
    
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let cell_size = 40;
    let grid_width = 5; // Fixed width for grid
    let grid_height = (filtered_words.len() + grid_width - 1) / grid_width;
    
    // Create a titled drawing area
    let root_area = root.titled("Word Frequency Heatmap", ("sans-serif", 40))?;
    let drawing_area = root_area.margin(20, 20, 20, 20);
    
    // Create grid for heatmap
    let mut heatmap_cells = Vec::new();
    for (i, (word, count)) in filtered_words.iter().enumerate() {
        let row = i / grid_width;
        let col = i % grid_width;
        
        let x = (col * cell_size) as i32;
        let y = (row * cell_size) as i32;
        
        // Calculate color intensity based on frequency
        let intensity = (*count as f64 / max_count as f64).powf(0.5); // Square root for better visualization
        
        // Use a color gradient from light blue to dark blue
        let color = HSLColor(
            0.6, // Hue for blue
            0.8, // Saturation 
            0.9 - intensity * 0.7, // Lightness from light to dark
        );
        
        heatmap_cells.push((x, y, word, *count, color));
    }
    
    for (x, y, word, count, color) in heatmap_cells {
        // Draw a rectangle for each word
        let cell = Rectangle::new(
            [(x, y), (x + cell_size as i32 - 2, y + cell_size as i32 - 2)],
            color.filled(),
        );
        
        drawing_area.draw(&cell)?;
        
        // Draw the word
        let text = Text::new(
            word.clone(),
            (x + cell_size as i32 / 2, y + cell_size as i32 / 2 - 5),
            ("sans-serif", (cell_size / 4).max(8) as f64),
        );
        drawing_area.draw(&text)?;
        
        // Draw the count
        let count_text = Text::new(
            count.to_string(),
            (x + cell_size as i32 / 2, y + cell_size as i32 / 2 + 10),
            ("sans-serif", (cell_size / 5).max(7) as f64),
        );
        drawing_area.draw(&count_text)?;
    }
    
    // Draw a legend for the heatmap
    let legend_width = 200;
    let legend_height = 20;
    let legend_x = 800;
    let legend_y = 30;
    
    // Draw gradient bar
    for i in 0..legend_width {
        let intensity = i as f64 / legend_width as f64;
        let color = HSLColor(
            0.6, // Hue for blue
            0.8, // Saturation
            0.9 - intensity * 0.7, // Lightness
        );
        
        let line = PathElement::new(
            vec![(legend_x + i, legend_y), (legend_x + i, legend_y + legend_height)],
            color.filled(),
        );
        root.draw(&line)?;
    }
    
    // Draw legend labels
    root.draw(&Text::new(
        "Less frequent",
        (legend_x, legend_y + legend_height + 15),
        ("sans-serif", 15),
    ))?;
    
    root.draw(&Text::new(
        "More frequent",
        (legend_x + legend_width - 100, legend_y + legend_height + 15),
        ("sans-serif", 15),
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_word_cloud(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Placeholder for a word cloud visualization
    // Creating a true word cloud requires complex algorithms for word placement
    // This is a simplified version that just displays words in different sizes
    
    // Filter out common stopwords
    let stopwords = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
    ];
    
    let filtered_words: Vec<_> = results.top_words
        .iter()
        .filter(|(word, _)| !stopwords.contains(&word.as_str()) && word.len() > 1)
        .take(limit)
        .collect();
    
    if filtered_words.is_empty() {
        return Err(anyhow::anyhow!("No words to visualize after filtering"));
    }
    
    let max_count = filtered_words.first().map(|(_, count)| *count).unwrap_or(0);
    let min_count = filtered_words.last().map(|(_, count)| *count).unwrap_or(0);
    
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let area = root.margin(20, 20, 20, 20);
    area.titled("Word Cloud", ("sans-serif", 50))?;
    
    // Simple grid layout for words
    let cols = (filtered_words.len() as f64).sqrt().ceil() as usize;
    let rows = (filtered_words.len() + cols - 1) / cols;
    
    let cell_width = 1200 / cols as i32;
    let cell_height = 700 / rows as i32;
    
    for (i, (word, count)) in filtered_words.iter().enumerate() {
        let row = i / cols;
        let col = i % cols;
        
        let x = col as i32 * cell_width + cell_width / 2;
        let y = row as i32 * cell_height + cell_height / 2 + 50; // 50px offset for title
        
        // Calculate font size based on frequency
        let size_ratio = (*count as f64 - min_count as f64) / (max_count as f64 - min_count as f64 + 1.0);
        let font_size = 15 + (55.0 * size_ratio) as u32;
        
        // Generate a consistent color based on the word itself
        let hue = (word.chars().fold(0, |acc, c| acc + c as u32) % 360) as f64 / 360.0;
        let color = HSLColor(hue, 0.7, 0.5);
        
        area.draw(&Text::new(
            word.clone(),
            (x, y),
            ("sans-serif", font_size).into_font().color(&color),
        ))?;
    }
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_ngram_chart(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Check if we have n-gram data
    let ngram_size = 3; // Default to trigrams
    
    let ngrams = if let Some(trigrams) = results.ngram_frequencies.get(&ngram_size) {
        trigrams
    } else if let Some(bigrams) = results.ngram_frequencies.get(&2) {
        bigrams
    } else {
        return Err(anyhow::anyhow!("No n-gram data available"));
    };
    
    let phrases: Vec<_> = ngrams.iter().take(limit.min(15)).collect();
    
    if phrases.is_empty() {
        return Err(anyhow::anyhow!("No phrases to visualize"));
    }
    
    let max_count = phrases.first().map(|(_, count)| *count).unwrap_or(0);
    
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(format!("Most Common {}-Grams", ngram_size), ("sans-serif", 40))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 200) // Extra space for longer phrases
        .build_cartesian_2d(
            0..phrases.len(),
            0..(max_count + max_count / 10),
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .y_desc("Frequency")
        .x_label_formatter(&|x| {
            if *x < phrases.len() {
                phrases[*x].0.clone()
            } else {
                String::new()
            }
        })
        .x_labels(phrases.len())
        .label_style(("sans-serif", 12))
        .x_label_style(("sans-serif", 12).into_font().transform(FontTransform::Rotate90))
        .draw()?;
    
    chart.draw_series(
        phrases.iter().enumerate().map(|(i, (_, count))| {
            let color = Palette99::pick(i).mix(0.9);
            Rectangle::new([(i, 0), (i + 1, *count)], color.filled())
        })
    )?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

pub fn generate_comparison_chart(
    results: &[AnalysisResults],
    output_path: &Path,
) -> Result<()> {
    if results.is_empty() {
        return Err(anyhow::anyhow!("No data to compare"));
    }
    
    // Prepare document names and colors
    let mut doc_names = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let name = if let Some(first_file) = result.file_stats.first() {
            // Simplify filename if it's too long
            if first_file.filename.len() > 20 {
                first_file.filename.chars().take(17).collect::<String>() + "..."
            } else {
                first_file.filename.clone()
            }
        } else {
            format!("Document {}", i + 1)
        };
        doc_names.push(name);
    }
    
    // Create root drawing area
    let root = BitMapBackend::new(output_path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Split into multiple chart areas
    let areas = root.split_evenly((2, 2));
    
    // 1. Word Count & Unique Words Chart
    let word_counts: Vec<_> = results.iter().map(|r| r.total_words as f64).collect();
    let unique_words: Vec<_> = results.iter().map(|r| r.unique_words as f64).collect();
    
    let max_words = *word_counts.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&1.0) * 1.1;
    
    let mut chart = ChartBuilder::on(&areas[0])
        .caption("Word Counts", ("sans-serif", 30))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            0.0..doc_names.len() as f64,
            0.0..max_words,
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .y_desc("Count")
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < doc_names.len() {
                doc_names[idx].clone()
            } else {
                String::new()
            }
        })
        .x_labels(doc_names.len())
        .label_style(("sans-serif", 12))
        .draw()?;
    
    // Draw total words bars
    chart.draw_series(
        word_counts.iter().enumerate().map(|(i, &count)| {
            Rectangle::new(
                [(i as f64 + 0.1, 0.0), (i as f64 + 0.4, count)],
                BLUE.mix(0.7).filled(),
            )
        })
    )?;
    
    // Draw unique words bars
    chart.draw_series(
        unique_words.iter().enumerate().map(|(i, &count)| {
            Rectangle::new(
                [(i as f64 + 0.5, 0.0), (i as f64 + 0.8, count)],
                RED.mix(0.7).filled(),
            )
        })
    )?;
    
    // Add a legend
    chart.configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(BLACK)
        .draw()?;
    
    chart.draw_series(std::iter::once(
        Rectangle::new([(doc_names.len() as f64 + 0.2, max_words * 0.9), 
                        (doc_names.len() as f64 + 0.4, max_words * 0.95)],
        BLUE.mix(0.7).filled())
    ))?.label("Total Words").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.mix(0.7).filled()));
    
    chart.draw_series(std::iter::once(
        Rectangle::new([(doc_names.len() as f64 + 0.2, max_words * 0.8), 
                        (doc_names.len() as f64 + 0.4, max_words * 0.85)],
        RED.mix(0.7).filled())
    ))?.label("Unique Words").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], RED.mix(0.7).filled()));
    
    // 2. Add a title to the chart
    root.draw(&Text::new(
        "Text Insights: Document Comparison",
        (600, 20),
        ("sans-serif", 40),
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
} 