.version 7.0
.target sm_70
.address_size 64

// --------------------
// compute_global_position
// Inputs: cam_x, cam_y, cam_heading, dist, off_x, off_y
// Writes 3 floats at ret_ptr: [global_x, global_y, global_z]
// --------------------
.visible .func compute_global_position(
    .param .f32 param_cam_x,
    .param .f32 param_cam_y,
    .param .f32 param_cam_heading,
    .param .f32 param_dist,
    .param .f32 param_off_x,
    .param .f32 param_off_y,
    .param .u64 param_ret_ptr
)
{
    .reg .f32  r_cam_x, r_cam_y, r_cam_heading;
    .reg .f32  r_dist, r_off_x, r_off_y;
    .reg .f32  r_angle, r_cos, r_sin;
    .reg .f32  r_global_x, r_global_y, r_global_z;
    .reg .u64  r_ret_ptr;

    ld.param.f32    r_cam_x, [param_cam_x];
    ld.param.f32    r_cam_y, [param_cam_y];
    ld.param.f32    r_cam_heading, [param_cam_heading];
    ld.param.f32    r_dist, [param_dist];
    ld.param.f32    r_off_x, [param_off_x];
    ld.param.f32    r_off_y, [param_off_y];
    ld.param.u64    r_ret_ptr, [param_ret_ptr];

    add.f32         r_angle, r_cam_heading, r_off_x;
    cos.approx.f32  r_cos, r_angle;
    sin.approx.f32  r_sin, r_angle;
    mul.f32         r_global_x, r_dist, r_cos;
    add.f32         r_global_x, r_cam_x, r_global_x;
    mul.f32         r_global_y, r_dist, r_sin;
    add.f32         r_global_y, r_cam_y, r_global_y;
    mov.f32         r_global_z, r_off_y;

    st.global.f32   [r_ret_ptr], r_global_x;
    st.global.f32   [r_ret_ptr+4], r_global_y;
    st.global.f32   [r_ret_ptr+8], r_global_z;
    ret;
}

// --------------------
// remove_duplicates for 2-float positions
// positions: array of 2-float entries; count: number of entries
// If distance < thresh then duplicate. Unique positions are stored in unique_ptr.
// unique_max is capacity; final unique count is written to unique_count_ptr.
// --------------------
.visible .func remove_duplicates(
    .param .u64 param_positions_ptr,
    .param .u32 param_count,
    .param .f32 param_thresh,
    .param .u64 param_unique_ptr,
    .param .u32 param_unique_max,
    .param .u64 param_unique_count_ptr
)
{
    .reg .u32 r_i, r_count, r_unique_count, r_unique_max, r_j;
    .reg .f32 f_thresh;
    .reg .u64 r_positions_ptr, r_unique_ptr, r_unique_count_ptr;

    ld.param.u64    r_positions_ptr, [param_positions_ptr];
    ld.param.u32    r_count, [param_count];
    ld.param.f32    f_thresh, [param_thresh];
    ld.param.u64    r_unique_ptr, [param_unique_ptr];
    ld.param.u32    r_unique_max, [param_unique_max];
    ld.param.u64    r_unique_count_ptr, [param_unique_count_ptr];

    mov.u32         r_unique_count, 0;
    mov.u32         r_i, 0;
LOOP_I:
    setp.ge.u32   p_end_i, r_i, r_count;
    @p_end_i      bra END_LOOP_I;
    .reg .u64     pos_i_addr;
    mul.lo.u32    pos_i_addr, r_i, 8;    // 2 floats * 4 bytes
    add.u64       pos_i_addr, r_positions_ptr, pos_i_addr;
    .reg .f32     pos_i_x, pos_i_y;
    ld.global.f32 pos_i_x, [pos_i_addr];
    ld.global.f32 pos_i_y, [pos_i_addr+4];
    .reg .u32     r_dup;
    mov.u32       r_dup, 0;
    mov.u32       r_j, 0;
LOOP_J:
    setp.ge.u32   p_end_j, r_j, r_unique_count;
    @p_end_j      bra AFTER_J;
    .reg .u64     unique_j_addr;
    mul.lo.u32    unique_j_addr, r_j, 8;
    add.u64       unique_j_addr, r_unique_ptr, unique_j_addr;
    .reg .f32     uniq_x, uniq_y;
    ld.global.f32 uniq_x, [unique_j_addr];
    ld.global.f32 uniq_y, [unique_j_addr+4];
    .reg .f32     diff_x, diff_y, sq_diff, sum_sq, dist;
    sub.f32       diff_x, pos_i_x, uniq_x;
    sub.f32       diff_y, pos_i_y, uniq_y;
    mul.f32       sq_diff, diff_x, diff_x;
    mul.f32       sum_sq, diff_y, diff_y;
    add.f32       sum_sq, sum_sq, sq_diff;
    sqrt.approx.f32 dist, sum_sq;
    setp.lt.f32   p_dup, dist, f_thresh;
    @p_dup      { mov.u32 r_dup, 1; bra END_J; }
    add.u32       r_j, r_j, 1;
    bra           LOOP_J;
END_J:
    setp.eq.u32   p_not_dup, r_dup, 0;
    @p_not_dup  {
        setp.ge.u32 p_full, r_unique_count, r_unique_max;
        @p_full    bra SKIP_STORE;
        .reg .u64 store_addr;
        mul.lo.u32 store_addr, r_unique_count, 8;
        add.u64 store_addr, r_unique_ptr, store_addr;
        st.global.f32 [store_addr], pos_i_x;
        st.global.f32 [store_addr+4], pos_i_y;
        add.u32 r_unique_count, r_unique_count, 1;
    }
SKIP_STORE:
    add.u32       r_i, r_i, 1;
    bra           LOOP_I;
END_LOOP_I:
    st.global.u32 [r_unique_count_ptr], r_unique_count;
    ret;
}

// --------------------
// remove_duplicates_3 for 3-float positions (compare only first 2)
// Each entry is 3 floats (12 bytes)
// --------------------
.visible .func remove_duplicates_3(
    .param .u64 param_positions_ptr,
    .param .u32 param_count,
    .param .f32 param_thresh,
    .param .u64 param_unique_ptr,
    .param .u32 param_unique_max,
    .param .u64 param_unique_count_ptr
)
{
    .reg .u32 r_i, r_count, r_unique_count, r_unique_max, r_j;
    .reg .f32 f_thresh;
    .reg .u64 r_positions_ptr, r_unique_ptr, r_unique_count_ptr;

    ld.param.u64    r_positions_ptr, [param_positions_ptr];
    ld.param.u32    r_count, [param_count];
    ld.param.f32    f_thresh, [param_thresh];
    ld.param.u64    r_unique_ptr, [param_unique_ptr];
    ld.param.u32    r_unique_max, [param_unique_max];
    ld.param.u64    r_unique_count_ptr, [param_unique_count_ptr];

    mov.u32         r_unique_count, 0;
    mov.u32         r_i, 0;
LOOP_I3:
    setp.ge.u32   p_end_i3, r_i, r_count;
    @p_end_i3     bra END_LOOP_I3;
    .reg .u64     pos_i_addr;
    mul.lo.u32    pos_i_addr, r_i, 12;
    add.u64       pos_i_addr, r_positions_ptr, pos_i_addr;
    .reg .f32     pos_i_x, pos_i_y;
    ld.global.f32 pos_i_x, [pos_i_addr];
    ld.global.f32 pos_i_y, [pos_i_addr+4];
    .reg .u32     r_dup;
    mov.u32       r_dup, 0;
    mov.u32       r_j, 0;
LOOP_J3:
    setp.ge.u32   p_end_j3, r_j, r_unique_count;
    @p_end_j3     bra AFTER_J3;
    .reg .u64     unique_j_addr;
    mul.lo.u32    unique_j_addr, r_j, 12;
    add.u64       unique_j_addr, r_unique_ptr, unique_j_addr;
    .reg .f32     uniq_x, uniq_y;
    ld.global.f32 uniq_x, [unique_j_addr];
    ld.global.f32 uniq_y, [unique_j_addr+4];
    .reg .f32     diff_x, diff_y, sq_diff, sum_sq, dist;
    sub.f32       diff_x, pos_i_x, uniq_x;
    sub.f32       diff_y, pos_i_y, uniq_y;
    mul.f32       sq_diff, diff_x, diff_x;
    mul.f32       sum_sq, diff_y, diff_y;
    add.f32       sum_sq, sum_sq, sq_diff;
    sqrt.approx.f32 dist, sum_sq;
    setp.lt.f32   p_dup3, dist, f_thresh;
    @p_dup3     { mov.u32 r_dup, 1; bra END_J3; }
    add.u32       r_j, r_j, 1;
    bra           LOOP_J3;
END_J3:
AFTER_J3:
    setp.eq.u32   p_not_dup3, r_dup, 0;
    @p_not_dup3 {
        setp.ge.u32 p_full3, r_unique_count, r_unique_max;
        @p_full3   bra SKIP_STORE3;
        .reg .u64 store_addr3;
        mul.lo.u32 store_addr3, r_unique_count, 12;
        add.u64 store_addr3, r_unique_ptr, store_addr3;
        ld.global.f32 pos_i_z, [pos_i_addr+8];
        st.global.f32 [store_addr3], pos_i_x;
        st.global.f32 [store_addr3+4], pos_i_y;
        st.global.f32 [store_addr3+8], pos_i_z;
        add.u32 r_unique_count, r_unique_count, 1;
    }
SKIP_STORE3:
    add.u32       r_i, r_i, 1;
    bra           LOOP_I3;
END_LOOP_I3:
    st.global.u32 [r_unique_count_ptr], r_unique_count;
    ret;
}

// --------------------
// localize_people kernel
// Inputs:
//   - cams_data: pointer to feed data (each feed has feed_length persons, 6 floats each)
//   - feed_count, feed_length
//   - grid_bounds: pointer to 4 floats [x_min,x_max,y_min,y_max]
//   - H, W: grid resolution
//   - kernel_size, sigma, dup_thresh
//   - heatmap_out: pointer to grid (H*W floats)
//   - temp_pos: temporary storage for (x,y) positions (max positions*2 floats)
//   - temp_pos_count_ptr: pointer to u32 count
// Performs: gather global positions, remove duplicates, accumulate density, and (optionally) convolve with a Gaussian.
// (Gaussian convolution loop omitted for brevity – full implementation would add nested loops.)
// --------------------
.entry localize_people(
    .param .u64 param_cams_data,
    .param .u32 param_feed_count,
    .param .u32 param_feed_length,
    .param .u64 param_grid_bounds_ptr,
    .param .u32 param_H,
    .param .u32 param_W,
    .param .u32 param_kernel_size,
    .param .f32 param_sigma,
    .param .f32 param_dup_thresh,
    .param .u64 param_heatmap_out,
    .param .u64 param_temp_pos,
    .param .u64 param_temp_pos_count_ptr
)
{
    .reg .u32 r_feed_count, r_feed_length, r_H, r_W, r_kernel_size;
    .reg .f32 f_sigma, f_dup_thresh;
    .reg .u64 r_cams_data, r_grid_bounds_ptr, r_heatmap_out, r_temp_pos, r_temp_pos_count_ptr;
    ld.param.u64    r_cams_data, [param_cams_data];
    ld.param.u32    r_feed_count, [param_feed_count];
    ld.param.u32    r_feed_length, [param_feed_length];
    ld.param.u64    r_grid_bounds_ptr, [param_grid_bounds_ptr];
    ld.param.u32    r_H, [param_H];
    ld.param.u32    r_W, [param_W];
    ld.param.u32    r_kernel_size, [param_kernel_size];
    ld.param.f32    f_sigma, [param_sigma];
    ld.param.f32    f_dup_thresh, [param_dup_thresh];
    ld.param.u64    r_heatmap_out, [param_heatmap_out];
    ld.param.u64    r_temp_pos, [param_temp_pos];
    ld.param.u64    r_temp_pos_count_ptr, [param_temp_pos_count_ptr];

    .reg .u32 r_pos_count;
    mov.u32         r_pos_count, 0;
    .reg .u32 r_i, r_j;
    mov.u32         r_i, 0;
FEED_LOOP:
    setp.ge.u32   p_feed_end, r_i, r_feed_count;
    @p_feed_end  bra END_FEED_LOOP;
    .reg .u32 r_feed_stride;
    mul.lo.u32    r_feed_stride, r_feed_length, 6;
    .reg .u64 feed_base;
    mul.wide.u32  feed_base, r_i, r_feed_stride;
    add.u64       feed_base, r_cams_data, feed_base;
    .reg .f32 cam_x, cam_y, cam_heading;
    ld.global.f32 cam_x, [feed_base];
    ld.global.f32 cam_y, [feed_base+4];
    ld.global.f32 cam_heading, [feed_base+8];
    mov.u32       r_j, 0;
PERSON_LOOP:
    setp.ge.u32   p_person_end, r_j, r_feed_length;
    @p_person_end bra END_PERSON_LOOP;
    .reg .u32 r_offset;
    mul.lo.u32    r_offset, r_j, 24;  // 6*4 bytes per person
    .reg .u64 person_ptr;
    add.u64       person_ptr, feed_base, r_offset;
    // Load peep info: dist at +12, off_x at +16, off_y at +20
    .reg .f32 dist, peep_off, off_y;
    ld.global.f32 dist, [person_ptr+12];
    ld.global.f32 peep_off, [person_ptr+16];
    ld.global.f32 off_y, [person_ptr+20];
    // Compute global pos: use compute_global_position inline
    .reg .f32 global_x, global_y, global_z;
    {
      .reg .f32 angle, cos_val, sin_val, temp;
      add.f32    angle, cam_heading, peep_off;
      cos.approx.f32 cos_val, angle;
      sin.approx.f32 sin_val, angle;
      mul.f32    temp, dist, cos_val;
      add.f32    global_x, cam_x, temp;
      mul.f32    temp, dist, sin_val;
      add.f32    global_y, cam_y, temp;
      mov.f32    global_z, off_y;
    }
    // Store (global_x, global_y) into temp_pos at index r_pos_count*2
    .reg .u32 r_temp_index;
    mov.u32      r_temp_index, r_pos_count;
    .reg .u64 pos_store_addr;
    mul.lo.u32   pos_store_addr, r_temp_index, 8;
    add.u64      pos_store_addr, r_temp_pos, pos_store_addr;
    st.global.f32 [pos_store_addr], global_x;
    st.global.f32 [pos_store_addr+4], global_y;
    add.u32      r_pos_count, r_pos_count, 1;
    add.u32      r_j, r_j, 1;
    bra          PERSON_LOOP;
END_PERSON_LOOP:
    add.u32      r_i, r_i, 1;
    bra          FEED_LOOP;
END_FEED_LOOP:
    st.global.u32 [r_temp_pos_count_ptr], r_pos_count;

    // Remove duplicates on temp_pos using remove_duplicates
    {
      .reg .u32 dummy;
      call (dummy), remove_duplicates, (r_temp_pos, r_pos_count, f_dup_thresh, r_temp_pos, r_pos_count, r_temp_pos_count_ptr);
    }

    // Zero heatmap array: total cells = H*W
    .reg .u32 r_index, r_total;
    mul.lo.u32    r_total, r_H, r_W;
    mov.u32       r_index, 0;
ZERO_HEATMAP:
    setp.ge.u32   p_zero_end, r_index, r_total;
    @p_zero_end  bra END_ZERO_HEATMAP;
    .reg .u64 heat_addr;
    mul.wide.u32  heat_addr, r_index, 4;
    add.u64       heat_addr, r_heatmap_out, heat_addr;
    st.global.f32 [heat_addr], 0f;
    add.u32       r_index, r_index, 1;
    bra           ZERO_HEATMAP;
END_ZERO_HEATMAP:

    // For each unique position, update heatmap cell
    .reg .u32 r_unique_count;
    ld.global.u32 r_unique_count, [r_temp_pos_count_ptr];
    mov.u32       r_index, 0;
UPDATE_HEATMAP:
    setp.ge.u32   p_update_end, r_index, r_unique_count;
    @p_update_end bra END_UPDATE_HEATMAP;
    .reg .u64 pos_addr;
    mul.lo.u32    pos_addr, r_index, 8;
    add.u64       pos_addr, r_temp_pos, pos_addr;
    .reg .f32 pos_x, pos_y;
    ld.global.f32 pos_x, [pos_addr];
    ld.global.f32 pos_y, [pos_addr+4];
    .reg .f32 x_min, x_max, y_min, y_max, diff_x, diff_y, norm_x, norm_y, f_temp;
    ld.global.f32 x_min, [r_grid_bounds_ptr];
    ld.global.f32 x_max, [r_grid_bounds_ptr+4];
    ld.global.f32 y_min, [r_grid_bounds_ptr+8];
    ld.global.f32 y_max, [r_grid_bounds_ptr+12];
    sub.f32 diff_x, x_max, x_min;
    sub.f32 diff_y, y_max, y_min;
    sub.f32 norm_x, pos_x, x_min;
    div.f32 norm_x, norm_x, diff_x;
    cvt.f32.u32 f_temp, r_W;  sub.f32 f_temp, f_temp, 1.0;
    mul.f32 norm_x, norm_x, f_temp;
    sub.f32 norm_y, pos_y, y_min;
    div.f32 norm_y, norm_y, diff_y;
    cvt.f32.u32 f_temp, r_H;  sub.f32 f_temp, f_temp, 1.0;
    mul.f32 norm_y, norm_y, f_temp;
    add.f32 norm_x, norm_x, 0.5;
    add.f32 norm_y, norm_y, 0.5;
    .reg .u32 grid_x, grid_y;
    cvt.rni.u32.f32 grid_x, norm_x;
    cvt.rni.u32.f32 grid_y, norm_y;
    setp.ge.u32 p_x_bound, grid_x, r_W;
    setp.ge.u32 p_y_bound, grid_y, r_H;
    or.pred p_out_bound, p_x_bound, p_y_bound;
    @p_out_bound bra SKIP_HEATMAP;
    .reg .u32 grid_index;
    mul.lo.u32 grid_index, grid_y, r_W;
    add.u32 grid_index, grid_index, grid_x;
    .reg .u64 heat_cell_addr;
    mul.wide.u32 heat_cell_addr, grid_index, 4;
    add.u64 heat_cell_addr, r_heatmap_out, heat_cell_addr;
    .reg .f32 cell_val;
    ld.global.f32 cell_val, [heat_cell_addr];
    add.f32 cell_val, cell_val, 1.0;
    st.global.f32 [heat_cell_addr], cell_val;
SKIP_HEATMAP:
    add.u32 r_index, r_index, 1;
    bra UPDATE_HEATMAP;
END_UPDATE_HEATMAP:

    // (Gaussian convolution code would go here.)
    ret;
}

// --------------------
// get_global_positions kernel
// Inputs similar to localize_people but stores full 3-float positions in global_out.
// Then removes duplicates (comparing only x,y).
// --------------------
.entry get_global_positions(
    .param .u64 param_cams_data,
    .param .u32 param_feed_count,
    .param .u32 param_feed_length,
    .param .u64 param_global_out,
    .param .u32 param_global_out_max,
    .param .u64 param_global_count_ptr,
    .param .f32 param_dup_thresh
)
{
    .reg .u32 r_feed_count, r_feed_length;
    .reg .u64 r_cams_data, r_global_out, r_global_count_ptr;
    .reg .u32 r_global_out_max;
    .reg .f32 f_dup_thresh;
    ld.param.u64    r_cams_data, [param_cams_data];
    ld.param.u32    r_feed_count, [param_feed_count];
    ld.param.u32    r_feed_length, [param_feed_length];
    ld.param.u64    r_global_out, [param_global_out];
    ld.param.u32    r_global_out_max, [param_global_out_max];
    ld.param.u64    r_global_count_ptr, [param_global_count_ptr];
    ld.param.f32    f_dup_thresh, [param_dup_thresh];

    .reg .u32 r_pos_count;
    mov.u32         r_pos_count, 0;
    .reg .u32 r_i, r_j;
    mov.u32         r_i, 0;
FEED_LOOP2:
    setp.ge.u32   p_feed_end2, r_i, r_feed_count;
    @p_feed_end2 bra END_FEED_LOOP2;
    .reg .u32 r_feed_stride;
    mul.lo.u32    r_feed_stride, r_feed_length, 6;
    .reg .u64 feed_base2;
    mul.wide.u32  feed_base2, r_i, r_feed_stride;
    add.u64       feed_base2, r_cams_data, feed_base2;
    .reg .f32 cam_x, cam_y, cam_heading;
    ld.global.f32 cam_x, [feed_base2];
    ld.global.f32 cam_y, [feed_base2+4];
    ld.global.f32 cam_heading, [feed_base2+8];
    mov.u32       r_j, 0;
PERSON_LOOP2:
    setp.ge.u32   p_person_end2, r_j, r_feed_length;
    @p_person_end2 bra END_PERSON_LOOP2;
    .reg .u32 r_offset2;
    mul.lo.u32    r_offset2, r_j, 24;
    .reg .u64 person_ptr2;
    add.u64       person_ptr2, feed_base2, r_offset2;
    .reg .f32 dist, off_actual, off_y;
    ld.global.f32 dist, [person_ptr2+12];
    ld.global.f32 off_actual, [person_ptr2+16];
    ld.global.f32 off_y, [person_ptr2+20];
    .reg .f32 global_x, global_y, global_z;
    {
       .reg .f32 angle, cos_val, sin_val, temp;
       add.f32    angle, cam_heading, off_actual;
       cos.approx.f32 cos_val, angle;
       sin.approx.f32 sin_val, angle;
       mul.f32    temp, dist, cos_val;
       add.f32    global_x, cam_x, temp;
       mul.f32    temp, dist, sin_val;
       add.f32    global_y, cam_y, temp;
       mov.f32    global_z, off_y;
    }
    .reg .u32 r_temp_index2;
    mov.u32      r_temp_index2, r_pos_count;
    .reg .u64 pos_store_addr2;
    mul.lo.u32   pos_store_addr2, r_temp_index2, 12;
    add.u64      pos_store_addr2, r_global_out, pos_store_addr2;
    st.global.f32 [pos_store_addr2], global_x;
    st.global.f32 [pos_store_addr2+4], global_y;
    st.global.f32 [pos_store_addr2+8], global_z;
    add.u32      r_pos_count, r_pos_count, 1;
    add.u32      r_j, r_j, 1;
    bra          PERSON_LOOP2;
END_PERSON_LOOP2:
    add.u32      r_i, r_i, 1;
    bra          FEED_LOOP2;
END_FEED_LOOP2:
    st.global.u32 [r_global_count_ptr], r_pos_count;
    {
       .reg .u32 dummy2;
       call (dummy2), remove_duplicates_3, (r_global_out, r_pos_count, f_dup_thresh, r_global_out, r_global_out_max, r_global_count_ptr);
    }
    ret;
}

.entry main
{
    ret;
}