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
        "parts-of-speech" => generate_pos_chart(results, output_path),
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
    let mut chars: Vec<_> = results.character_frequencies.iter().collect();
    chars.sort_by(|(_, count1), (_, count2)| count2.cmp(count1));
    
    let chars_to_display = chars.into_iter().take(limit.min(30)).collect::<Vec<_>>();
    
    if chars_to_display.is_empty() {
        return Err(anyhow::anyhow!("No character data to visualize"));
    }
    
    // Create a larger, higher-resolution canvas
    let root = BitMapBackend::new(output_path, (1600, 900)).into_drawing_area();
    
    // Use a subtle gradient background
    root.fill(&RGBColor(245, 247, 250))?;
    
    // Create a titled chart with clear, attractive styling
    let mut chart = ChartBuilder::on(&root)
        .margin(60)
        .caption(
            "CHARACTER FREQUENCY DISTRIBUTION", 
            ("sans-serif", 46).into_font().color(&RGBColor(40, 45, 98))
        )
        .set_label_area_size(LabelAreaPosition::Left, 80)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .build_cartesian_2d(
            0..chars_to_display.len() as i32,
            0..(chars_to_display.iter().map(|(_, count)| **count).max().unwrap_or(0) as i32) + 5,
        )?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .light_line_style(RGBColor(240, 240, 245))
        .bold_line_style(RGBColor(220, 220, 230))
        .axis_style(ShapeStyle::from(&RGBColor(100, 100, 100)).stroke_width(2))
        .y_desc("FREQUENCY")
        .y_label_style(("sans-serif", 26).into_font().color(&RGBColor(60, 60, 80)))
        .x_label_formatter(&|x| {
            if *x < chars_to_display.len() as i32 {
                // Special formatting for whitespace and control characters
                let ch = chars_to_display[*x as usize].0;
                match ch {
                    ' ' => "SPACE".to_string(),
                    '\t' => "TAB".to_string(),
                    '\n' => "NEWLINE".to_string(),
                    '\r' => "CR".to_string(),
                    ch if ch.is_control() => format!("CTL-{:02X}", *ch as u8),
                    _ => ch.to_string(),
                }
            } else {
                String::new()
            }
        })
        .x_labels(chars_to_display.len())
        .label_style(("sans-serif", 20).into_font().color(&RGBColor(60, 60, 80)))
        .draw()?;
    
    // Add horizontal gridlines
    let max_count = chars_to_display.iter().map(|(_, count)| **count).max().unwrap_or(0);
    for y in (0..=max_count).step_by((max_count / 10).max(1) as usize) {
        if y > 0 {
            chart.draw_series(std::iter::once(
                PathElement::new(
                    vec![(0, y as i32), (chars_to_display.len() as i32, y as i32)],
                    RGBColor(235, 235, 245).stroke_width(1)
                )
            ))?;
        }
    }
    
    // Use a visually appealing gradient palette
    // Each character gets its own distinctive color but follows a pleasing gradient
    let base_hue = 210.0 / 360.0; // Blue base
    
    // Draw bars with distinctive coloring and clear spacing
    chart.draw_series(
        chars_to_display.iter().enumerate().map(|(i, (ch, count))| {
            // Each character gets a related but distinct color
            let hue = (base_hue + (i as f64 * 0.8 / chars_to_display.len() as f64)) % 1.0;
            let color = HSLColor(hue, 0.7, 0.5).mix(0.9);
            
            // Calculate bar width with proper spacing
            let bar_width = 0.7; // Width of each bar (out of 1.0 unit)
            let x0 = (i as f64 + (1.0 - bar_width) / 2.0) as i32;
            let x1 = (i as f64 + (1.0 - bar_width) / 2.0 + bar_width) as i32;
            
            // Use a rectangle with rounded corners for a modern look
            Rectangle::new(
                [(x0, 0), (x1, **count as i32)],
                color.filled(),
            )
        })
    )?;
    
    // Add the count numbers on top of each bar for clarity
    for (i, (_, count)) in chars_to_display.iter().enumerate() {
        let count_str = count.to_string();
        chart.draw_series(std::iter::once(
            Text::new(
                count_str,
                (i as i32, **count as i32 + max_count as i32 / 40),
                ("sans-serif", 20).into_font().color(&RGBColor(60, 60, 100))
            )
        ))?;
    }
    
    // Add source info at the top
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        root.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (800, 30),
            ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100))
        ))?;
    }
    
    // Add a summary caption at the bottom
    let total_chars = results.character_frequencies.values().sum::<usize>();
    let unique_chars = results.character_frequencies.len();
    
    let footer_text = format!(
        "Analysis of {} unique characters from {} total characters", 
        unique_chars, 
        total_chars
    );
    
    root.draw(&Text::new(
        footer_text,
        (800, 850),
        ("sans-serif", 24).into_font().color(&RGBColor(80, 80, 100))
    ))?;
    
    // Add generation date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    root.draw(&Text::new(
        format!("Generated: {}", current_date),
        (1400, 850),
        ("sans-serif", 16).into_font().color(&RGBColor(150, 150, 170))
    ))?;
    
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
    let sentiment = results.sentiment_score.unwrap_or(0.0);
    
    // Create a high-resolution canvas
    let root = BitMapBackend::new(output_path, (1200, 900)).into_drawing_area();
    
    // Use a light gradient background for better aesthetics
    root.fill(&RGBColor(250, 252, 255))?;
    
    let drawing_area = root.margin(30, 30, 30, 30);
    
    // Draw a bold title with a subtitle
    drawing_area.draw(&Text::new(
        "SENTIMENT ANALYSIS",
        (600, 60),
        ("sans-serif", 50).into_font().color(&RGBColor(30, 30, 80)),
    ))?;
    
    // Add file information
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        drawing_area.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (600, 120),
            ("sans-serif", 22).into_font().color(&RGBColor(80, 80, 100)),
        ))?;
    }
    
    // Calculate a normalized sentiment score (0-1 range)
    let normalized_sentiment = (sentiment + 1.0) / 2.0;
    
    // Draw a modern gauge with gradient
    let gauge_center_x = 600;
    let gauge_center_y = 450;
    let gauge_radius = 250;
    let gauge_thickness = 50;
    
    // Draw a gradient background track for the gauge
    let start_angle = -140_f64.to_radians();
    let end_angle = -40_f64.to_radians();
    let angle_range = end_angle - start_angle;
    
    // Draw the background track with a gradient from light to dark
    let steps = 100;
    for i in 0..steps {
        let start_step = start_angle + (i as f64 / steps as f64) * angle_range;
        let end_step = start_angle + ((i + 1) as f64 / steps as f64) * angle_range;
        
        let outer_radius = gauge_radius + gauge_thickness / 2;
        let inner_radius = gauge_radius - gauge_thickness / 2;
        
        let outer_start = (
            gauge_center_x + (outer_radius as f64 * start_step.cos()) as i32,
            gauge_center_y + (outer_radius as f64 * start_step.sin()) as i32
        );
        let outer_end = (
            gauge_center_x + (outer_radius as f64 * end_step.cos()) as i32,
            gauge_center_y + (outer_radius as f64 * end_step.sin()) as i32
        );
        let inner_end = (
            gauge_center_x + (inner_radius as f64 * end_step.cos()) as i32,
            gauge_center_y + (inner_radius as f64 * end_step.sin()) as i32
        );
        let inner_start = (
            gauge_center_x + (inner_radius as f64 * start_step.cos()) as i32,
            gauge_center_y + (inner_radius as f64 * start_step.sin()) as i32
        );
        
        // Calculate color based on position (red to green gradient)
        let position = i as f64 / steps as f64;
        let color = if position < 0.5 {
            // Red to yellow gradient for negative sentiment
            let mix = position * 2.0;
            RGBColor(
                200 + (55.0 * mix) as u8,
                (200.0 * mix) as u8,
                50,
            )
        } else {
            // Yellow to green gradient for positive sentiment
            let mix = (position - 0.5) * 2.0;
            RGBColor(
                255 - (200.0 * mix) as u8,
                200 + (55.0 * mix) as u8,
                50,
            )
        };
        
        drawing_area.draw(&Polygon::new(
            vec![outer_start, outer_end, inner_end, inner_start],
            &color.mix(0.7),
        ))?;
    }
    
    // Draw tick marks and labels
    for i in 0..=4 {
        let tick_position = i as f64 / 4.0;
        let tick_angle = start_angle + tick_position * angle_range;
        
        let outer_tick = (
            gauge_center_x + ((gauge_radius + gauge_thickness / 2 + 10) as f64 * tick_angle.cos()) as i32,
            gauge_center_y + ((gauge_radius + gauge_thickness / 2 + 10) as f64 * tick_angle.sin()) as i32
        );
        
        let label_position = (
            gauge_center_x + ((gauge_radius + gauge_thickness / 2 + 40) as f64 * tick_angle.cos()) as i32,
            gauge_center_y + ((gauge_radius + gauge_thickness / 2 + 40) as f64 * tick_angle.sin()) as i32
        );
        
        // Calculate the sentiment value at this tick position (-1 to 1)
        let sentiment_value = -1.0 + tick_position * 2.0;
        
        // Determine the label based on sentiment value
        let label = match sentiment_value {
            x if x <= -0.8 => "Very Negative",
            x if x <= -0.3 => "Negative",
            x if x <= 0.3 => "Neutral",
            x if x <= 0.8 => "Positive",
            _ => "Very Positive",
        };
        
        drawing_area.draw(&Text::new(
            label,
            label_position,
            ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100)),
        ))?;
    }
    
    // Draw the needle
    let needle_angle = start_angle + normalized_sentiment * angle_range;
    let needle_length = gauge_radius - 20;
    let needle_width = 8;
    
    // Calculate the needle points
    let needle_tip = (
        gauge_center_x + (needle_length as f64 * needle_angle.cos()) as i32,
        gauge_center_y + (needle_length as f64 * needle_angle.sin()) as i32
    );
    
    let perpendicular_angle = needle_angle + std::f64::consts::PI / 2.0;
    let needle_base_left = (
        gauge_center_x + (needle_width as f64 * perpendicular_angle.cos()) as i32,
        gauge_center_y + (needle_width as f64 * perpendicular_angle.sin()) as i32
    );
    
    let needle_base_right = (
        gauge_center_x - (needle_width as f64 * perpendicular_angle.cos()) as i32,
        gauge_center_y - (needle_width as f64 * perpendicular_angle.sin()) as i32
    );
    
    // Draw the needle with a gradient
    drawing_area.draw(&Polygon::new(
        vec![needle_tip, needle_base_left, needle_base_right],
        &RGBColor(180, 30, 30).mix(0.9),
    ))?;
    
    // Draw a center cap for the needle
    drawing_area.draw(&Circle::new(
        (gauge_center_x, gauge_center_y),
        15,
        RGBColor(100, 100, 110).filled(),
    ))?;
    
    drawing_area.draw(&Circle::new(
        (gauge_center_x, gauge_center_y),
        12,
        RGBColor(180, 180, 200).filled(),
    ))?;
    
    // Display the exact sentiment score
    let score_label = format!("Sentiment Score: {:.2}", sentiment);
    drawing_area.draw(&Text::new(
        score_label,
        (600, gauge_center_y + gauge_radius + 100),
        ("sans-serif", 36).into_font().color(&RGBColor(50, 50, 80)),
    ))?;
    
    // Add interpretative context based on the sentiment
    let context = match sentiment {
        x if x <= -0.8 => "The text displays strongly negative emotions or opinions.",
        x if x <= -0.3 => "The text contains more negative than positive elements.",
        x if x <= 0.3 => "The text is relatively neutral in sentiment.",
        x if x <= 0.8 => "The text contains more positive than negative elements.",
        _ => "The text displays strongly positive emotions or opinions.",
    };
    
    drawing_area.draw(&Text::new(
        context,
        (600, gauge_center_y + gauge_radius + 150),
        ("sans-serif", 24).into_font().color(&RGBColor(80, 80, 100)),
    ))?;
    
    // Add document stats
    let stats_text = format!(
        "Analysis of {} words across {} sentences", 
        results.total_words,
        results.sentence_count
    );
    
    drawing_area.draw(&Text::new(
        stats_text,
        (600, 800),
        ("sans-serif", 20).into_font().color(&RGBColor(100, 100, 120)),
    ))?;
    
    // Add generation date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    drawing_area.draw(&Text::new(
        format!("Generated: {}", current_date),
        (600, 840),
        ("sans-serif", 16).into_font().color(&RGBColor(150, 150, 170)),
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

fn generate_word_heatmap(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Filter out stopwords for better visualization
    let stopwords = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
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
    
    // Create a larger canvas for better visualization
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    
    // Use a subtle gradient background for better aesthetics
    root.fill(&RGBColor(250, 250, 255))?;
    
    // Add title and metadata
    let title_area = root.titled(
        "WORD FREQUENCY HEATMAP",
        ("sans-serif", 60)
            .into_font()
            .color(&RGBColor(40, 40, 100))
    )?;
    
    // Add source information
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        root.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (800, 100),
            ("sans-serif", 24).into_font().color(&RGBColor(80, 80, 100)),
        ))?;
    }
    
    // Set up a better grid layout with optimal cell size
    let grid_dimensions = calculate_grid_dimensions(filtered_words.len());
    let grid_width = grid_dimensions.0;
    let grid_height = grid_dimensions.1;
    
    // Calculate cell size based on available space
    let available_width = 1500;
    let available_height = 900;
    let cell_size = (available_width / grid_width).min(available_height / grid_height);
    
    // Create drawing area with margins
    let drawing_area = title_area.margin(50, 50, 150, 50);
    
    // Store cells to be drawn
    let mut heatmap_cells = Vec::new();
    
    // Use a more sophisticated color gradient for better visual differentiation
    for (i, (word, count)) in filtered_words.iter().enumerate() {
        let row = i / grid_width;
        let col = i % grid_width;
        
        let x = (col * cell_size) as i32 + 50; // Add offset for better margin
        let y = (row * cell_size) as i32 + 150; // Add offset to account for title
        
        // Calculate color using a more sophisticated approach
        // Use a logarithmic scale for better visual distribution
        let intensity = ((*count as f64) / (max_count as f64)).powf(0.4); // Power of 0.4 gives better distribution
        
        // Create a color based on frequency - use a blue to purple gradient
        let hue = 0.6 + intensity * 0.2; // Range from blue (0.6) to purple (0.8)
        let saturation = 0.7 + intensity * 0.3; // More frequent = more saturated
        let lightness = 0.9 - intensity * 0.6; // More frequent = darker
        
        let color = HSLColor(hue, saturation, lightness);
        
        heatmap_cells.push((x, y, word, *count, color, intensity));
    }
    
    // Draw the cells
    for (x, y, word, count, color, intensity) in heatmap_cells {
        // Create a rectangle with rounded corners for modern look
        let cell = Rectangle::new(
            [(x, y), (x + cell_size as i32 - 4, y + cell_size as i32 - 4)],
            color.filled(),
        );
        
        // Add a subtle border with shadow effect
        let border = Rectangle::new(
            [(x, y), (x + cell_size as i32 - 4, y + cell_size as i32 - 4)],
            RGBColor(100, 100, 140).mix(0.2 + intensity * 0.3).stroke_width(1),
        );
        
        drawing_area.draw(&cell)?;
        drawing_area.draw(&border)?;
        
        // Calculate font size based on cell size and word length
        let max_word_length = 12;
        let word_length_factor = (max_word_length.min(word.len()) as f64 / max_word_length as f64).powf(0.5);
        let base_font_size = (cell_size as f64 / 5.0) * (1.0 - word_length_factor * 0.3);
        let font_size = base_font_size.max(9.0).min(24.0) as u32;
        
        // Draw the word with contrasting color for readability
        // Make the text color adapt to the background brightness
        let text_color = if lightness_value(&color) > 0.6 {
            // Dark text on light background
            RGBColor(20, 20, 60)
        } else {
            // Light text on dark background
            RGBColor(240, 240, 255)
        };
        
        // Draw the word
        drawing_area.draw(&Text::new(
            word.clone(),
            (x + cell_size as i32 / 2, y + cell_size as i32 / 2 - font_size as i32 / 2),
            ("sans-serif", font_size).into_font().color(&text_color),
        ))?;
        
        // Draw the count with a slightly different color
        drawing_area.draw(&Text::new(
            count.to_string(),
            (x + cell_size as i32 / 2, y + cell_size as i32 / 2 + font_size as i32),
            ("sans-serif", (font_size as f64 * 0.9) as u32).into_font().color(&text_color.mix(0.9)),
        ))?;
    }
    
    // Draw a more sophisticated legend for the heatmap
    let legend_width = 300;
    let legend_height = 30;
    let legend_x = 1200;
    let legend_y = 1050;
    
    // Draw legend title
    root.draw(&Text::new(
        "Frequency Legend",
        (legend_x + legend_width / 2, legend_y - 30),
        ("sans-serif", 24).into_font().color(&RGBColor(60, 60, 80)),
    ))?;
    
    // Draw gradient bar with segments
    let segments = 60;
    for i in 0..segments {
        let intensity = i as f64 / segments as f64;
        
        let hue = 0.6 + intensity * 0.2;
        let saturation = 0.7 + intensity * 0.3;
        let lightness = 0.9 - intensity * 0.6;
        let color = HSLColor(hue, saturation, lightness);
        
        let segment_width = legend_width / segments;
        let segment_x = legend_x + i * segment_width;
        
        let rect = Rectangle::new(
            [(segment_x, legend_y), (segment_x + segment_width, legend_y + legend_height)],
            color.filled(),
        );
        
        root.draw(&rect)?;
    }
    
    // Add tick marks and labels
    for i in 0..=5 {
        let position = i as f64 / 5.0;
        let tick_x = legend_x + (position * legend_width as f64) as i32;
        
        // Draw tick mark
        root.draw(&PathElement::new(
            vec![(tick_x, legend_y + legend_height), (tick_x, legend_y + legend_height + 10)],
            RGBColor(60, 60, 80).stroke_width(2),
        ))?;
        
        // Frequency label - show as percentage
        let label = if i == 0 {
            "Least frequent".to_string()
        } else if i == 5 {
            "Most frequent".to_string()
        } else {
            format!("{}%", i * 20)
        };
        
        root.draw(&Text::new(
            label,
            (tick_x, legend_y + legend_height + 25),
            ("sans-serif", 16).into_font().color(&RGBColor(60, 60, 80)),
        ))?;
    }
    
    // Add summary information
    let footer_text = format!(
        "This heatmap shows the {} most frequent words from a total of {} unique words",
        filtered_words.len(),
        results.unique_words
    );
    
    root.draw(&Text::new(
        footer_text,
        (800, 1120),
        ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100)),
    ))?;
    
    // Add generation date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    root.draw(&Text::new(
        format!("Generated: {}", current_date),
        (1400, 1120),
        ("sans-serif", 16).into_font().color(&RGBColor(150, 150, 170)),
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
}

// Helper function to calculate grid dimensions
fn calculate_grid_dimensions(item_count: usize) -> (usize, usize) {
    let sqrt = (item_count as f64).sqrt().ceil() as usize;
    
    // Aim for a grid that's slightly wider than tall for better viewing on most screens
    let cols = sqrt;
    let rows = (item_count + cols - 1) / cols; // Ceiling division
    
    // If rows is much smaller than cols, redistribute
    if (rows as f64) * 1.5 < cols as f64 {
        let new_cols = (item_count as f64 / (item_count as f64 / cols as f64 * 1.3).ceil()).ceil() as usize;
        let new_rows = (item_count + new_cols - 1) / new_cols;
        (new_cols, new_rows)
    } else {
        (cols, rows)
    }
}

// Helper function to calculate perceived lightness of a color
fn lightness_value(color: &HSLColor) -> f64 {
    // HSLColor contains hue, saturation, lightness
    // Just extract the lightness component which is what we need
    color.2 // This is the lightness component
}

fn generate_word_cloud(
    results: &AnalysisResults,
    output_path: &Path,
    limit: usize,
) -> Result<()> {
    // Filter out common stopwords
    let stopwords = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
        "could", "may", "might", "must", "shall",
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
    
    // Create a larger canvas for better word placement
    let root = BitMapBackend::new(output_path, (1600, 1000)).into_drawing_area();
    
    // Use a gradient background for more visual appeal
    root.fill(&BLUE.mix(0.05))?;
    
    let area = root.margin(40, 40, 40, 40);
    
    // Create a modern title with a subtitle
    area.draw(&Text::new(
        "WORD CLOUD",
        (800, 60),
        ("sans-serif", 60).into_font().color(&BLUE.mix(0.8)),
    ))?;
    
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        area.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (800, 120),
            ("sans-serif", 24).into_font().color(&BLACK.mix(0.6)),
        ))?;
    }
    
    // Use a more sophisticated layout algorithm for word placement
    // This is a spiral layout algorithm that places words in a spiral pattern
    let center_x = 800;
    let center_y = 500;
    let mut spiral_angle: f64 = 0.0;
    let mut spiral_radius: f64 = 50.0;
    let spiral_step: f64 = 0.5;
    
    // Keep track of used positions to avoid overlaps
    let mut used_positions = Vec::new();
    
    for (i, (word, count)) in filtered_words.iter().enumerate() {
        // Calculate font size based on frequency
        let size_ratio = (*count as f64 - min_count as f64) / (max_count as f64 - min_count as f64 + 1.0);
        let font_size = 18.0 + (65.0 * size_ratio);
        
        // Generate a consistent color based on the word itself
        // Using a better color palette with more variation
        let hue = ((i * 31) % 360) as f64 / 360.0; // Prime number 31 gives better distribution
        let saturation = 0.7 + (size_ratio * 0.3); // More frequent = more saturated
        let lightness = 0.4 + (1.0 - size_ratio) * 0.3; // More frequent = darker
        let color = HSLColor(hue, saturation, lightness);
        
        // Find a position for this word using the spiral
        let mut x = 0;
        let mut y = 0;
        let mut found_position = false;
        let word_width = (word.len() as f64 * font_size * 0.6) as i32;
        let word_height = font_size as i32;
        
        // Try up to 500 positions along the spiral
        for _ in 0..500 {
            x = center_x + (spiral_radius * spiral_angle.cos()) as i32;
            y = center_y + (spiral_radius * spiral_angle.sin()) as i32;
            
            // Check if position overlaps with any existing word
            let overlaps = used_positions.iter().any(|(px, py, pw, ph)| {
                x - word_width/2 < px + pw/2 &&
                x + word_width/2 > px - pw/2 &&
                y - word_height/2 < py + ph/2 &&
                y + word_height/2 > py - ph/2
            });
            
            if !overlaps {
                found_position = true;
                break;
            }
            
            // Move along the spiral
            spiral_angle += spiral_step;
            spiral_radius += spiral_step * 0.5;
        }
        
        if !found_position {
            // If we can't find a good spot, just place it somewhere
            x = ((i % 5) * 300 + 150) as i32;
            y = ((i / 5) * 150 + 300) as i32;
        }
        
        // Record this word's position
        used_positions.push((x, y, word_width, word_height));
        
        // Add a subtle drop shadow for depth
        area.draw(&Text::new(
            word.clone(),
            (x + 2, y + 2),
            ("sans-serif", font_size as u32).into_font().color(&BLACK.mix(0.2)),
        ))?;
        
        // Draw the word
        area.draw(&Text::new(
            word.clone(),
            (x, y),
            ("sans-serif", font_size as u32).into_font().color(&color),
        ))?;
    }
    
    // Add a legend to show frequency correlation
    let legend_y = 900;
    area.draw(&Text::new(
        "Word size indicates frequency",
        (800, legend_y),
        ("sans-serif", 20).into_font().color(&BLACK.mix(0.7)),
    ))?;
    
    // Add the date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    area.draw(&Text::new(
        format!("Generated: {}", current_date),
        (1400, legend_y),
        ("sans-serif", 16).into_font().color(&BLACK.mix(0.5)),
    ))?;
    
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

fn generate_pos_chart(
    results: &AnalysisResults,
    output_path: &Path,
) -> Result<()> {
    // Check if we have POS data
    if results.pos_counts.is_empty() {
        return Err(anyhow::anyhow!("No part-of-speech data available"));
    }
    
    // Create a beautiful, high-resolution canvas
    let root = BitMapBackend::new(output_path, (1200, 900)).into_drawing_area();
    
    // Use a subtle gradient background
    root.fill(&RGBColor(250, 252, 255))?;
    
    // Add a title and subtitle
    root.draw(&Text::new(
        "PARTS OF SPEECH ANALYSIS",
        (600, 50),
        ("sans-serif", 50).into_font().color(&RGBColor(40, 40, 100)),
    ))?;
    
    // Add source information
    if let Some(first_file) = results.file_stats.first() {
        let filename = first_file.filename.clone();
        let truncated_name = if filename.len() > 40 {
            format!("{}...", &filename[..37])
        } else {
            filename
        };
        
        root.draw(&Text::new(
            format!("Source: {}", truncated_name),
            (600, 100),
            ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100)),
        ))?;
    }
    
    // Prepare the POS data
    let mut pos_data: Vec<_> = results.pos_counts.iter().collect();
    pos_data.sort_by(|(_, count1), (_, count2)| count2.cmp(count1));
    
    let total_pos = pos_data.iter().map(|(_, count)| **count).sum::<usize>();
    
    // Create a pie chart area
    let drawing_area = root.margin(80, 80, 80, 80);
    
    // Define the center and radius of the pie chart
    let center = (500, 400);
    let radius = 250;
    
    // Define friendly names for POS tags
    let pos_friendly_names = |tag: &str| -> String {
        match tag {
            "NOUN" => "Nouns".to_string(),
            "VERB" => "Verbs".to_string(),
            "ADJ" => "Adjectives".to_string(),
            "ADV" => "Adverbs".to_string(),
            "PRON" => "Pronouns".to_string(),
            "DET" => "Determiners".to_string(),
            "ADP" => "Prepositions".to_string(),
            "CONJ" => "Conjunctions".to_string(),
            "PART" => "Particles".to_string(),
            "NUM" => "Numerals".to_string(),
            "PUNCT" => "Punctuation".to_string(),
            "SYM" => "Symbols".to_string(),
            "INTJ" => "Interjections".to_string(),
            _ => tag.to_string(),
        }
    };
    
    // Define a beautiful color palette
    let colors = vec![
        RGBColor(41, 121, 255),  // Blue
        RGBColor(255, 99, 71),   // Tomato
        RGBColor(50, 205, 50),   // Lime Green
        RGBColor(255, 165, 0),   // Orange
        RGBColor(138, 43, 226),  // Blue Violet
        RGBColor(0, 139, 139),   // Dark Cyan
        RGBColor(255, 20, 147),  // Deep Pink
        RGBColor(0, 191, 255),   // Deep Sky Blue
        RGBColor(255, 215, 0),   // Gold
        RGBColor(154, 205, 50),  // Yellow Green
        RGBColor(219, 112, 147), // Pale Violet Red
        RGBColor(95, 158, 160),  // Cadet Blue
        RGBColor(255, 127, 80),  // Coral
        RGBColor(106, 90, 205),  // Slate Blue
        RGBColor(173, 255, 47),  // Green Yellow
    ];
    
    // Draw pie chart slices
    let mut current_angle = 0.0;
    let mut legend_items = Vec::new();
    
    for (i, (pos, count)) in pos_data.iter().enumerate() {
        let percentage = **count as f64 / total_pos as f64;
        let angle = percentage * 2.0 * std::f64::consts::PI;
        
        // Skip tiny slices (less than 1%)
        if percentage < 0.01 {
            continue;
        }
        
        // Calculate slice points
        let end_angle = current_angle + angle;
        
        // Generate points for the slice
        let mut points = Vec::new();
        points.push(center);
        
        // Add points along the arc
        let steps = (angle * 30.0).ceil() as usize;
        for j in 0..=steps {
            let step_angle = current_angle + (j as f64 / steps as f64) * angle;
            let x = center.0 + (radius as f64 * step_angle.cos()) as i32;
            let y = center.1 + (radius as f64 * step_angle.sin()) as i32;
            points.push((x, y));
        }
        
        // Draw the slice
        let color_idx = i % colors.len();
        let color = colors[color_idx].mix(0.8);
        
        drawing_area.draw(&Polygon::new(points.clone(), color.filled()))?;
        drawing_area.draw(&Polygon::new(points, colors[color_idx].stroke_width(1)))?;
        
        // Calculate position for slice label
        let label_angle = current_angle + angle / 2.0;
        let label_distance = radius as f64 * 0.7; // 70% of the way to edge
        let label_x = center.0 + (label_distance * label_angle.cos()) as i32;
        let label_y = center.1 + (label_distance * label_angle.sin()) as i32;
        
        // Only show label in the slice if percentage is large enough
        if percentage > 0.05 {
            drawing_area.draw(&Text::new(
                format!("{:.1}%", percentage * 100.0),
                (label_x, label_y),
                ("sans-serif", 20).into_font().color(&WHITE),
            ))?;
        }
        
        // Store information for legend
        legend_items.push((pos, **count, percentage, color));
        
        // Update current angle
        current_angle = end_angle;
    }
    
    // Draw legend
    let legend_x = 800;
    let legend_y = 250;
    let legend_spacing = 35;
    
    root.draw(&Text::new(
        "LEGEND:",
        (legend_x, legend_y - legend_spacing),
        ("sans-serif", 24).into_font().color(&RGBColor(60, 60, 100)),
    ))?;
    
    for (i, (pos, count, percentage, color)) in legend_items.iter().enumerate() {
        let y_pos = legend_y + i as i32 * legend_spacing;
        
        // Draw colored square
        root.draw(&Rectangle::new(
            [(legend_x, y_pos - 15), (legend_x + 20, y_pos + 5)],
            color.filled(),
        ))?;
        
        // Draw text
        root.draw(&Text::new(
            format!("{} - {}: {} ({:.1}%)", 
                i + 1, 
                pos_friendly_names(pos), 
                count, 
                percentage * 100.0
            ),
            (legend_x + 30, y_pos),
            ("sans-serif", 18).into_font().color(&RGBColor(60, 60, 100)),
        ))?;
    }
    
    // Add overall statistics
    let unique_pos = results.pos_counts.len();
    
    let stats_text = format!(
        "Analysis found {} different parts of speech across {} words", 
        unique_pos,
        results.total_words
    );
    
    root.draw(&Text::new(
        stats_text,
        (600, 700),
        ("sans-serif", 20).into_font().color(&RGBColor(80, 80, 100)),
    ))?;
    
    // Add explanatory text
    root.draw(&Text::new(
        "A balanced text typically uses a variety of parts of speech.",
        (600, 740),
        ("sans-serif", 18).into_font().color(&RGBColor(100, 100, 130)),
    ))?;
    
    // Add generation date
    let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
    root.draw(&Text::new(
        format!("Generated: {}", current_date),
        (600, 780),
        ("sans-serif", 16).into_font().color(&RGBColor(150, 150, 170)),
    ))?;
    
    root.present().context("Failed to write image to file")?;
    
    Ok(())
} 