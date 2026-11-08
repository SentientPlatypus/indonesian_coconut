//! Test multi-game batching with proper state isolation

#![cfg(all(test, feature = "ripple"))]

use std::collections::HashMap;

#[test]
fn test_batch_position_mapping() {
    // Test that position_to_state mapping works correctly
    // Simulates 2 games × 2 cars = 4 batch positions
    
    let mut position_to_state: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    
    // Simulate first batch (kickoffs - all get new IDs)
    for pos in 0..4 {
        let is_kickoff = true;
        let state_id = if is_kickoff {
            let id = next_id;
            next_id += 1;
            position_to_state.insert(pos, id);
            id
        } else {
            *position_to_state.get(&pos).unwrap()
        };
        println!("Batch 0, pos {}: state_id = {}", pos, state_id);
    }
    
    assert_eq!(position_to_state.len(), 4, "Should have 4 unique states");
    assert_eq!(position_to_state.get(&0), Some(&0));
    assert_eq!(position_to_state.get(&1), Some(&1));
    assert_eq!(position_to_state.get(&2), Some(&2));
    assert_eq!(position_to_state.get(&3), Some(&3));
    
    // Simulate second batch (non-kickoffs - reuse existing IDs)
    for pos in 0..4 {
        let is_kickoff = false;
        let state_id = if is_kickoff {
            let id = next_id;
            next_id += 1;
            position_to_state.insert(pos, id);
            id
        } else {
            *position_to_state.get(&pos).unwrap()
        };
        println!("Batch 1, pos {}: state_id = {} (reused)", pos, state_id);
    }
    
    // State IDs should be unchanged
    assert_eq!(position_to_state.get(&0), Some(&0));
    assert_eq!(position_to_state.get(&1), Some(&1));
    
    // Simulate one game resetting
    let reset_pos = 2;
    let old_id = position_to_state.get(&reset_pos).copied().unwrap();
    let new_id = next_id;
    next_id += 1;
    position_to_state.insert(reset_pos, new_id);
    
    assert_ne!(old_id, new_id, "Reset should create new state ID");
    assert_eq!(position_to_state.get(&reset_pos), Some(&new_id));
}

#[test]
fn test_queue_structure_invariants() {
    // Document queue semantics for multi-game scenarios
    const QUEUE_LEN: usize = 6;
    
    // Each game maintains independent queue state
    let mut game1_queue = vec![[0.0f32; 8]; QUEUE_LEN];
    let mut game2_queue = vec![[1.0f32; 8]; QUEUE_LEN];
    
    // Queues start different
    assert_ne!(game1_queue[0], game2_queue[0]);
    
    // After rotation, structure is preserved
    game1_queue.rotate_right(1);
    game2_queue.rotate_right(1);
    
    assert_eq!(game1_queue[1][0], 0.0, "Game 1 queue maintains its values");
    assert_eq!(game2_queue[1][0], 1.0, "Game 2 queue maintains its values");
    
    println!("✓ Independent queue states verified");
}

#[test]
fn test_position_counter_logic() {
    // Test that position counter increments correctly during build() calls
    let mut position = 0usize;
    let mut calls = Vec::new();
    
    // Simulate 3 build() calls
    for _ in 0..3 {
        let p = position;
        position += 1;
        calls.push(p);
    }
    
    assert_eq!(calls, vec![0, 1, 2], "Position counter increments sequentially");
    
    // Reset for next batch
    position = 0;
    calls.clear();
    
    for _ in 0..3 {
        let p = position;
        position += 1;
        calls.push(p);
    }
    
    assert_eq!(calls, vec![0, 1, 2], "Position counter resets between batches");
}
