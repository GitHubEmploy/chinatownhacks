.version 7.0
.target sm_70
.address_size 64

// --------------------------------------------------------------------------
// Device function: rand_uniform
// Updates a state in global memory (a u32) and returns a float in [0,1)
// --------------------------------------------------------------------------
.visible .func rand_uniform(
    .param .u64 param_state_ptr,
    .param .u64 param_out_ptr
)
{
    .reg .u32 s, new_s;
    .reg .f32 f;
    .reg .u64 state_ptr, out_ptr;
    ld.param.u64   state_ptr, [param_state_ptr];
    ld.param.u64   out_ptr, [param_out_ptr];

    ld.global.u32  s, [state_ptr];
    mul.lo.u32     new_s, s, 1664525;
    add.u32        new_s, new_s, 1013904223;
    st.global.u32  [state_ptr], new_s;
    cvt.f32.u32    f, new_s;
    // Scale by 1/4294967296.0 ~ 2.328306e-10
    mul.f32        f, f, 2.328306e-10;
    st.global.f32  [out_ptr], f;
    ret;
}

// --------------------------------------------------------------------------
// Device function: rand_gaussian
// Uses Box-Muller to compute a gaussian random number with given mean and std.
// The state pointer is updated internally.
// --------------------------------------------------------------------------
.visible .func rand_gaussian(
    .param .u64 param_state_ptr,
    .param .f32 param_mean,
    .param .f32 param_std,
    .param .u64 param_out_ptr
)
{
    .reg .u64 state_ptr, out_ptr;
    .reg .f32 u1, u2, log_u1, sqrt_term, angle, cos_val, gauss;
    .reg .f32 mean, std;
    ld.param.u64   state_ptr, [param_state_ptr];
    ld.param.u64   out_ptr, [param_out_ptr];
    ld.param.f32   mean, [param_mean];
    ld.param.f32   std, [param_std];

    // Get first uniform random number u1
    {
      .reg .u32 s, new_s;
      .reg .u64 tmp_ptr;
      mov.u64 tmp_ptr, state_ptr;
      call ( ) rand_uniform, ( tmp_ptr, out_ptr );
      ld.global.f32 u1, [out_ptr];
    }
    // Get second uniform random number u2
    {
      .reg .u32 s2, new_s2;
      .reg .u64 tmp_ptr2;
      mov.u64 tmp_ptr2, state_ptr;
      call ( ) rand_uniform, ( tmp_ptr2, out_ptr );
      ld.global.f32 u2, [out_ptr];
    }
    // Compute sqrt_term = sqrt(-2 * log(u1))
    mov.f32       log_u1, 0.0;  // reuse log_u1 register
    log.approx.f32 log_u1, u1;
    mul.f32       log_u1, log_u1, -2.0;
    sqrt.approx.f32 sqrt_term, log_u1;
    // Compute angle = 2*pi*u2 (2*pi ~ 6.2831853)
    mov.f32       angle, 6.2831853;
    mul.f32       angle, angle, u2;
    cos.approx.f32 cos_val, angle;
    mul.f32       gauss, sqrt_term, cos_val;
    // Apply mean and std: result = mean + std * gauss
    mul.f32       gauss, std, gauss;
    add.f32       gauss, mean, gauss;
    st.global.f32 [out_ptr], gauss;
    ret;
}

// --------------------------------------------------------------------------
// Kernel: generate_sample_data
// Simulates detection feeds from multiple cameras and stores detections.
// Each detection is 7 floats: [cam_x, cam_y, cam_heading, det_distance,
//  det_off_x, det_off_y, store_id].
// The simulated_detections buffer must be sized as NUM_CAMERAS*MAX_DETECTIONS*7
// and detection_counts is an array of NUM_CAMERAS u32 elements.
// Global arrays (global_cameras, global_stores, global_store_phases) must be
// arranged as described.
// --------------------------------------------------------------------------
.entry generate_sample_data(
    .param .u64 param_global_cameras,       // float[NUM_CAMERAS*3]
    .param .u64 param_global_stores,          // float[NUM_STORES*2]
    .param .u64 param_global_store_phases,    // float[NUM_STORES]
    .param .f32 param_simulation_time,        // current simulation time
    .param .u64 param_seed_ptr,               // pointer to u32 seed
    .param .u64 param_simulated_detections,   // output float array
    .param .u64 param_detection_counts        // output u32 array
)
{
    .reg .u64 r_global_cameras, r_global_stores, r_global_store_phases;
    .reg .f32 sim_time;
    .reg .u64 r_seed_ptr, r_simulated_detections, r_detection_counts;
    ld.param.u64   r_global_cameras, [param_global_cameras];
    ld.param.u64   r_global_stores, [param_global_stores];
    ld.param.u64   r_global_store_phases, [param_global_store_phases];
    ld.param.f32   sim_time, [param_simulation_time];
    ld.param.u64   r_seed_ptr, [param_seed_ptr];
    ld.param.u64   r_simulated_detections, [param_simulated_detections];
    ld.param.u64   r_detection_counts, [param_detection_counts];

    // Constants
    .reg .u32 NUM_CAMERAS, NUM_STORES, MAX_DETECTIONS;
    mov.u32        NUM_CAMERAS, 6;
    mov.u32        NUM_STORES, 5;
    mov.u32        MAX_DETECTIONS, 256;
    .reg .f32 MIN_GROUP, MAX_GROUP, PERIOD, SIM_DELTA, PI, HALF_PI, TWENTY, TWO_PI;
    mov.f32        MIN_GROUP, 8.0;
    mov.f32        MAX_GROUP, 20.0;
    mov.f32        PERIOD, 10.0;
    mov.f32        SIM_DELTA, 0.05;
    mov.f32        PI, 3.14159265;
    mov.f32        HALF_PI, 1.57079633;
    mov.f32        TWENTY, 20.0;
    mov.f32        TWO_PI, 6.2831853;

    // Loop over cameras: for (i=0; i<NUM_CAMERAS; i++)
    .reg .u32 i;
    mov.u32        i, 0;
CAM_LOOP:
    setp.ge.u32   p_cam_end, i, NUM_CAMERAS;
    @p_cam_end    bra END_CAM_LOOP;

    // Compute feed pointer: feed = simulated_detections + i*MAX_DETECTIONS*7*sizeof(f32)
    .reg .u32 feed_stride;
    mul.lo.u32    feed_stride, MAX_DETECTIONS, 7;
    .reg .u32 feed_index;
    mul.lo.u32    feed_index, i, feed_stride;
    .reg .u64 feed_byte_offset;
    mul.wide.u32  feed_byte_offset, feed_index, 4;
    .reg .u64 feed_ptr;
    add.u64       feed_ptr, r_simulated_detections, feed_byte_offset;

    // Set detection count for camera i to 0.
    .reg .u32 det_count;
    mov.u32       det_count, 0;
    .reg .u64 count_offset;
    mul.wide.u32  count_offset, i, 4;
    .reg .u64 count_ptr;
    add.u64       count_ptr, r_detection_counts, count_offset;
    st.global.u32 [count_ptr], 0;

    // Load camera info: each camera is 3 floats, index = i*3.
    .reg .u32 cam_offset;
    mul.lo.u32    cam_offset, i, 3;
    .reg .u64 cam_byte_offset;
    mul.wide.u32  cam_byte_offset, cam_offset, 4;
    .reg .u64 cam_ptr;
    add.u64       cam_ptr, r_global_cameras, cam_byte_offset;
    .reg .f32 cam_x, cam_y, cam_heading;
    ld.global.f32 cam_x, [cam_ptr];
    ld.global.f32 cam_y, [cam_ptr+4];
    ld.global.f32 cam_heading, [cam_ptr+8];

    // Loop over stores: for (sid=0; sid<NUM_STORES; sid++)
    .reg .u32 sid;
    mov.u32       sid, 0;
STORE_LOOP:
    setp.ge.u32   p_store_end, sid, NUM_STORES;
    @p_store_end bra END_STORE_LOOP;

    // Load store info: each store has 2 floats.
    .reg .u32 store_offset;
    mul.lo.u32    store_offset, sid, 2;
    .reg .u64 store_byte_offset;
    mul.wide.u32  store_byte_offset, store_offset, 4;
    .reg .u64 store_ptr;
    add.u64       store_ptr, r_global_stores, store_byte_offset;
    .reg .f32 store_x, store_y;
    ld.global.f32 store_x, [store_ptr];
    ld.global.f32 store_y, [store_ptr+4];

    // dx = store_x - cam_x; dy = store_y - cam_y; distance = sqrt(dx^2+dy^2)
    .reg .f32 dx, dy, dx2, dy2, sum_sq, distance;
    sub.f32       dx, store_x, cam_x;
    sub.f32       dy, store_y, cam_y;
    mul.f32       dx2, dx, dx;
    mul.f32       dy2, dy, dy;
    add.f32       sum_sq, dx2, dy2;
    sqrt.approx.f32 distance, sum_sq;

    // angle_to_store = atan2(dy, dx)
    .reg .f32 angle_to_store;
    atan2.approx.f32 angle_to_store, dy, dx;

    // rel_angle = angle_to_store - cam_heading
    .reg .f32 rel_angle;
    sub.f32       rel_angle, angle_to_store, cam_heading;

    // Normalize rel_angle to [-PI, PI]
NORM_LOOP1:
    setp.gt.f32 p_norm1, rel_angle, PI;
    @p_norm1 {
        sub.f32   rel_angle, rel_angle, TWO_PI;
        bra       NORM_LOOP1;
    }
NORM_LOOP2:
    setp.lt.f32 p_norm2, rel_angle, -PI;
    @p_norm2 {
        add.f32   rel_angle, rel_angle, TWO_PI;
        bra       NORM_LOOP2;
    }

    // If (abs(rel_angle) < HALF_PI and distance < TWENTY) then add group detections.
    .reg .f32 abs_rel;
    // Compute absolute value (using conditional move)
    setp.lt.f32 p_tmp, rel_angle, 0.0;
    @p_tmp { neg.f32 abs_rel, rel_angle; }
    @!p_tmp { mov.f32 abs_rel, rel_angle; }
    setp.lt.f32 p_cond1, abs_rel, HALF_PI;
    setp.lt.f32 p_cond2, distance, TWENTY;
    and.pred   p_valid, p_cond1, p_cond2;
    @p_valid {
        // Load phase from global_store_phases: each store phase is one float.
        .reg .u32 phase_offset;
        mov.u32   phase_offset, sid;
        .reg .u64 phase_byte_offset;
        mul.wide.u32 phase_byte_offset, phase_offset, 4;
        .reg .u64 phase_ptr;
        add.u64   phase_ptr, r_global_store_phases, phase_byte_offset;
        .reg .f32 phase;
        ld.global.f32 phase, [phase_ptr];

        // activity = sin(2*PI*(sim_time/PERIOD + phase)); if negative, set to 0.
        .reg .f32 ratio, sum_val, arg, activity;
        div.f32   ratio, sim_time, PERIOD;
        add.f32   sum_val, ratio, phase;
        mul.f32   arg, TWO_PI, sum_val;
        sin.approx.f32 activity, arg;
        setp.lt.f32 p_act, activity, 0.0;
        @p_act   mov.f32 activity, 0.0;
        // group_size = MIN_GROUP + (MAX_GROUP - MIN_GROUP)*activity, rounded to nearest int.
        .reg .f32 group_f;
        sub.f32   group_f, MAX_GROUP, MIN_GROUP;
        mul.f32   group_f, group_f, activity;
        add.f32   group_f, group_f, MIN_GROUP;
        .reg .u32 group_size;
        cvt.rni.u32.f32 group_size, group_f;

        // Loop for group detections: for (j=0; j<group_size; j++)
        .reg .u32 j;
        mov.u32   j, 0;
GROUP_LOOP:
        setp.ge.u32 p_group_end, j, group_size;
        @p_group_end bra END_GROUP_LOOP;

        // det_distance = distance + rand_gaussian(state, 0, 0.2)
        .reg .f32 noise_dist, det_distance;
        {
          .reg .u32 dummy;
          .reg .f32 m_val, s_val;
          mov.f32 m_val, 0.0;
          mov.f32 s_val, 0.2;
          call (dummy), rand_gaussian, ( r_seed_ptr, m_val, s_val, r_seed_ptr );
          ld.global.f32 noise_dist, [r_seed_ptr]; // reuse r_seed_ptr as temp storage
        }
        add.f32   det_distance, distance, noise_dist;

        // det_off_x = rel_angle + rand_gaussian(state, 0, 0.05)
        .reg .f32 noise_off, det_off_x;
        {
          .reg .u32 dummy2;
          .reg .f32 m_val2, s_val2;
          mov.f32 m_val2, 0.0;
          mov.f32 s_val2, 0.05;
          call (dummy2), rand_gaussian, ( r_seed_ptr, m_val2, s_val2, r_seed_ptr );
          ld.global.f32 noise_off, [r_seed_ptr];
        }
        add.f32   det_off_x, rel_angle, noise_off;
        // det_off_y = 0.0
        .reg .f32 det_off_y;
        mov.f32   det_off_y, 0.0;

        // Write detection: [cam_x, cam_y, cam_heading, det_distance, det_off_x, det_off_y, store_id]
        .reg .u32 det_index;
        mov.u32   det_index, det_count;
        .reg .u32 det_offset;
        mul.lo.u32 det_offset, det_index, 7;
        .reg .u64 det_byte_offset;
        mul.wide.u32 det_byte_offset, det_offset, 4;
        .reg .u64 det_ptr;
        add.u64   det_ptr, feed_ptr, det_byte_offset;
        st.global.f32 [det_ptr], cam_x;
        st.global.f32 [det_ptr+4], cam_y;
        st.global.f32 [det_ptr+8], cam_heading;
        st.global.f32 [det_ptr+12], det_distance;
        st.global.f32 [det_ptr+16], det_off_x;
        st.global.f32 [det_ptr+20], det_off_y;
        // Write store id as float (sid converted to float)
        cvt.f32.u32 stemp, sid;
        st.global.f32 [det_ptr+24], stemp;
        add.u32   det_count, det_count, 1;

        add.u32   j, j, 1;
        bra       GROUP_LOOP;
END_GROUP_LOOP:
    }

    // Extra detections: extra = (rand_uniform()*4) floored; here we simulate extra = 0.
    .reg .u32 extra;
    mov.u32   extra, 0;
    .reg .u32 k;
    mov.u32   k, 0;
EXTRA_LOOP:
    setp.ge.u32 p_extra_end, k, extra;
    @p_extra_end bra END_EXTRA_LOOP;
    // det_distance = rand_gaussian(state, 5, 1)
    .reg .f32 extra_dist;
    {
      .reg .u32 dummy3;
      .reg .f32 m_val3, s_val3;
      mov.f32 m_val3, 5.0;
      mov.f32 s_val3, 1.0;
      call (dummy3), rand_gaussian, ( r_seed_ptr, m_val3, s_val3, r_seed_ptr );
      ld.global.f32 extra_dist, [r_seed_ptr];
    }
    // det_off_x = rand_gaussian(state, 0, 0.5)
    .reg .f32 extra_off;
    {
      .reg .u32 dummy4;
      .reg .f32 m_val4, s_val4;
      mov.f32 m_val4, 0.0;
      mov.f32 s_val4, 0.5;
      call (dummy4), rand_gaussian, ( r_seed_ptr, m_val4, s_val4, r_seed_ptr );
      ld.global.f32 extra_off, [r_seed_ptr];
    }
    .reg .f32 extra_off_y;
    mov.f32   extra_off_y, 0.0;
    // Write extra detection with store_id = -1.0
    .reg .u32 extra_index;
    mov.u32   extra_index, det_count;
    .reg .u32 extra_offset;
    mul.lo.u32 extra_offset, extra_index, 7;
    .reg .u64 extra_byte_offset;
    mul.wide.u32 extra_byte_offset, extra_offset, 4;
    .reg .u64 extra_ptr;
    add.u64   extra_ptr, feed_ptr, extra_byte_offset;
    st.global.f32 [extra_ptr], cam_x;
    st.global.f32 [extra_ptr+4], cam_y;
    st.global.f32 [extra_ptr+8], cam_heading;
    st.global.f32 [extra_ptr+12], extra_dist;
    st.global.f32 [extra_ptr+16], extra_off;
    st.global.f32 [extra_ptr+20], extra_off_y;
    st.global.f32 [extra_ptr+24], -1.0;
    add.u32   det_count, det_count, 1;
    add.u32   k, k, 1;
    bra       EXTRA_LOOP;
END_EXTRA_LOOP:

    // Write the final detection count for camera i.
    st.global.u32 [count_ptr], det_count;

    add.u32   i, i, 1;
    bra       CAM_LOOP;
END_CAM_LOOP:
    ret;
}

// --------------------------------------------------------------------------
// Kernel: compute_accuracy
// Given an array of global_positions (each 3 floats) of length 'count',
// grid bounds (4 floats: x_min,x_max,y_min,y_max) and grid resolution (H,W),
// compute the average positional error and return an accuracy percentage via
// param_accuracy_out.
// --------------------------------------------------------------------------
.entry compute_accuracy(
    .param .u64 param_global_positions, // float array of [x,y,_]
    .param .u32 param_count,
    .param .u64 param_bounds,           // float[4]: x_min, x_max, y_min, y_max
    .param .u32 param_H,
    .param .u32 param_W,
    .param .u64 param_accuracy_out      // pointer to a float
)
{
    .reg .u64 r_positions, r_bounds, r_accuracy_out;
    .reg .u32 count, H, W;
    ld.param.u64   r_positions, [param_global_positions];
    ld.param.u32   count, [param_count];
    ld.param.u64   r_bounds, [param_bounds];
    ld.param.u32   H, [param_H];
    ld.param.u32   W, [param_W];
    ld.param.u64   r_accuracy_out, [param_accuracy_out];

    // Load bounds: x_min, x_max, y_min, y_max
    .reg .f32 x_min, x_max, y_min, y_max;
    ld.global.f32  x_min, [r_bounds];
    ld.global.f32  x_max, [r_bounds+4];
    ld.global.f32  y_min, [r_bounds+8];
    ld.global.f32  y_max, [r_bounds+12];

    // Compute cell widths cw and ch.
    .reg .f32 cw, ch;
    sub.f32   cw, x_max, x_min;
    sub.f32   ch, y_max, y_min;
    cvt.f32.u32 cw, W;  div.f32 cw, cw, cw; // Overwrite cw with cw = (x_max-x_min)/W
    sub.f32   cw, x_max, x_min; div.f32 cw, cw, ( (f32)W );
    sub.f32   ch, y_max, y_min; div.f32 ch, ch, ( (f32)H );
    // For simplicity, we precompute max_error = sqrt((cw/2)^2+(ch/2)^2)
    .reg .f32 half_cw, half_ch, sq1, sq2, max_error;
    mul.f32   half_cw, cw, 0.5;
    mul.f32   half_ch, ch, 0.5;
    mul.f32   sq1, half_cw, half_cw;
    mul.f32   sq2, half_ch, half_ch;
    add.f32   max_error, sq1, sq2;
    sqrt.approx.f32 max_error, max_error;

    // Loop over each position and accumulate error.
    .reg .u32 i;
    mov.u32   i, 0;
    .reg .f32 total_err;
    mov.f32   total_err, 0.0;
LOOP_POS:
    setp.ge.u32 p_pos_end, i, count;
    @p_pos_end bra END_LOOP_POS;
    // Each position is 3 floats (12 bytes); load x, y.
    .reg .u32 pos_offset;
    mul.lo.u32 pos_offset, i, 3;
    .reg .u64 pos_byte_offset;
    mul.wide.u32 pos_byte_offset, pos_offset, 4;
    .reg .u64 pos_ptr;
    add.u64   pos_ptr, r_positions, pos_byte_offset;
    .reg .f32 pos_x, pos_y;
    ld.global.f32 pos_x, [pos_ptr];
    ld.global.f32 pos_y, [pos_ptr+4];
    // Compute grid indices i_idx and j_idx (rounded).
    .reg .f32 norm_x, norm_y, grid_x_f, grid_y_f;
    sub.f32   norm_x, pos_x, x_min;
    sub.f32   norm_y, pos_y, y_min;
    div.f32   norm_x, norm_x, (x_max - x_min);
    div.f32   norm_y, norm_y, (y_max - y_min);
    mul.f32   grid_x_f, norm_x, ( (f32)(W-1) );
    mul.f32   grid_y_f, norm_y, ( (f32)(H-1) );
    cvt.rni.u32.f32 grid_x, grid_x_f;
    cvt.rni.u32.f32 grid_y, grid_y_f;
    // Compute center of cell.
    .reg .f32 cx, cy, err;
    // For simplicity, assume center = x_min + (i+0.5)*cw, similarly for y.
    cvt.f32.u32 cx, grid_x;  // reuse register
    mul.f32   cx, cx, cw;
    add.f32   cx, cx, x_min;
    add.f32   cx, cx, 0.5 * cw;
    cvt.f32.u32 cy, grid_y;
    mul.f32   cy, cy, ch;
    add.f32   cy, cy, y_min;
    add.f32   cy, cy, 0.5 * ch;
    sub.f32   err, pos_x, cx;
    mul.f32   err, err, err;
    .reg .f32 temp;
    sub.f32   temp, pos_y, cy;
    mul.f32   temp, temp, temp;
    add.f32   err, err, temp;
    sqrt.approx.f32 err, err;
    add.f32   total_err, total_err, err;
    add.u32   i, i, 1;
    bra       LOOP_POS;
END_LOOP_POS:
    // Average error = total_err / count
    .reg .f32 avg_err;
    cvt.f32.u32 avg_err, count;
    div.f32   avg_err, total_err, avg_err;
    // Compute accuracy = max(0, (1 - avg_err/max_error)*100)
    .reg .f32 acc;
    div.f32   acc, avg_err, max_error;
    sub.f32   acc, 1.0, acc;
    mul.f32   acc, acc, 100.0;
    setp.lt.f32 p_neg, acc, 0.0;
    @p_neg    mov.f32 acc, 0.0;
    st.global.f32 [r_accuracy_out], acc;
    ret;
}

// --------------------------------------------------------------------------
// Main entry stub (host must launch kernels and handle I/O)
// --------------------------------------------------------------------------
.entry main
{
    ret;
}