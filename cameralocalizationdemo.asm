.version 6.0
.target sm_50
.address_size 64

// KRN1: Compute global positions for each detection.
// IN:  in_feed_ptr -> arr of recs (6 floats each):
//         [cam_x, cam_y, cam_heading, det_dist, det_off_x, det_off_y]
//      num_records -> # of recs
// OUT: out_positions_ptr -> arr of recs (3 floats each):
//         [glob_x, glob_y, glob_z]
.visible .entry compute_global_positions_kernel(
    .param .u64 in_feed_ptr,
    .param .u64 out_positions_ptr,
    .param .u32 num_records
)
{
    // Thread idx calc.
    .reg .s32   %tid, %ctaid, %ntid, %idx, %offset;
    // FP regs for cam & det data.
    .reg .f32   %cam_x, %cam_y, %cam_heading;
    .reg .f32   %dist, %off_x, %off_y;
    .reg .f32   %glob_angle, %f_sin, %f_cos;
    .reg .f32   %glob_x, %glob_y, %glob_z;
    // Ptr regs.
    .reg .u64   %in_ptr, %out_ptr, %rec_ptr, %store_ptr;
    .reg .s32   %rec_offset;
    .pred      %p;

    // Compute global thread index.
    mov.u32    %tid, %tid.x;
    mov.u32    %ctaid, %ctaid.x;
    mov.u32    %ntid, %ntid.x;
    mad.lo.s32 %idx, %ctaid, %ntid, %tid;

    // Exit if idx >= num_records.
    ld.param.u32   %offset, [num_records];
    setp.ge.s32    %p, %idx, %offset;
    @%p bra       EXIT0;

    // Load ptrs from params.
    ld.param.u64   %in_ptr, [in_feed_ptr];
    ld.param.u64   %out_ptr, [out_positions_ptr];

    // Each rec = 6 floats (24 bytes); calc addr for current rec.
    mul.lo.s32   %rec_offset, %idx, 24;
    cvt.u64.s32  %rec_ptr, %rec_offset;
    add.u64      %rec_ptr, %in_ptr, %rec_ptr;

    // Load detection rec.
    ld.global.f32  %cam_x,      [%rec_ptr];       // offset 0
    ld.global.f32  %cam_y,      [%rec_ptr+4];     // offset 4
    ld.global.f32  %cam_heading,[%rec_ptr+8];     // offset 8
    ld.global.f32  %dist,       [%rec_ptr+12];    // offset 12
    ld.global.f32  %off_x,      [%rec_ptr+16];    // offset 16
    ld.global.f32  %off_y,      [%rec_ptr+20];    // offset 20

    // Calc glob_angle = cam_heading + det_off_x.
    add.f32    %glob_angle, %cam_heading, %off_x;

    // Compute sin & cos (approx).
    sin.approx.f32   %f_sin, %glob_angle;
    cos.approx.f32   %f_cos, %glob_angle;

    // Calc glob_x = cam_x + dist * cos(glob_angle).
    mul.f32    %glob_x, %dist, %f_cos;
    add.f32    %glob_x, %cam_x, %glob_x;

    // Calc glob_y = cam_y + dist * sin(glob_angle).
    mul.f32    %glob_y, %dist, %f_sin;
    add.f32    %glob_y, %cam_y, %glob_y;

    // glob_z = det_off_y.
    mov.f32   %glob_z, %off_y;

    // Store result in out_positions_ptr (each rec = 3 floats = 12 bytes).
    mul.lo.s32   %rec_offset, %idx, 12;
    cvt.u64.s32  %store_ptr, %rec_offset;
    add.u64      %store_ptr, %out_ptr, %store_ptr;
    st.global.f32  [%store_ptr],      %glob_x;
    st.global.f32  [%store_ptr+4],    %glob_y;
    st.global.f32  [%store_ptr+8],    %glob_z;

EXIT0:
    ret;
}

// KRN2: Accumulate density from positions.
// IN:  positions_ptr -> arr of recs (2 floats each): [x, y]
//      num_positions -> # of pos recs
//      density_ptr -> ptr to density grid (flat arr of floats)
//      x_min, x_max, y_min, y_max -> grid bounds (FP)
//      grid_W -> grid width (u32)
// Normalizes (x,y) to grid indices & uses atomics to add.
.visible .entry accumulate_density_kernel(
    .param .u64 positions_ptr,
    .param .u32 num_positions,
    .param .u64 density_ptr,
    .param .f32 x_min,
    .param .f32 x_max,
    .param .f32 y_min,
    .param .f32 y_max,
    .param .u32 grid_W
)
{
    // Thread idx calc.
    .reg .s32   %tid, %ctaid, %ntid, %idx, %rec_offset;
    .reg .f32   %x, %y;
    .reg .f32   %xs_norm, %ys_norm;
    .reg .s32   %x_idx, %y_idx;
    .reg .u64   %pos_ptr, %density_base, %offset_ptr;
    .reg .f32   %temp;
    .reg .s32   %gridWidth, %prod, %num;
    .pred      %p;

    // Compute global thread idx.
    mov.u32    %tid, %tid.x;
    mov.u32    %ctaid, %ctaid.x;
    mov.u32    %ntid, %ntid.x;
    mad.lo.s32 %idx, %ctaid, %ntid, %tid;

    ld.param.u32   %num, [num_positions];
    setp.ge.s32    %p, %idx, %num;
    @%p bra       EXIT1;

    ld.param.u64   %pos_ptr, [positions_ptr];
    // Each pos rec = 2 floats (8 bytes); calc addr for current pos.
    mul.lo.s32   %rec_offset, %idx, 8;
    cvt.u64.s32  %rec_offset, %rec_offset;
    add.u64      %pos_ptr, %pos_ptr, %rec_offset;

    // Load x & y.
    ld.global.f32  %x, [%pos_ptr];
    ld.global.f32  %y, [%pos_ptr+4];

    // Normalize x -> grid idx.
    ld.param.f32  %temp, [x_min];
    sub.f32       %xs_norm, %x, %temp;         // x - x_min
    ld.param.f32  %temp, [x_max];
    sub.f32       %temp, %temp, %x_min;          // (x_max - x_min)
    div.f32       %xs_norm, %xs_norm, %temp;
    ld.param.u32  %gridWidth, [grid_W];
    cvt.f32.u32   %temp, %gridWidth;
    sub.f32       %temp, %temp, 1.0;
    mul.f32       %xs_norm, %xs_norm, %temp;
    cvt.rni.f32.s32 %x_idx, %xs_norm;

    // Normalize y -> grid idx.
    ld.param.f32  %temp, [y_min];
    sub.f32       %ys_norm, %y, %temp;         // y - y_min
    ld.param.f32  %temp, [y_max];
    sub.f32       %temp, %temp, %y_min;          // (y_max - y_min)
    div.f32       %ys_norm, %ys_norm, %temp;
    cvt.rni.f32.s32 %y_idx, %ys_norm;

    // Get base ptr for density grid.
    ld.param.u64  %density_base, [density_ptr];
    // Compute flat idx: idx = y_idx * grid_W + x_idx.
    mul.lo.s32  %prod, %y_idx, %gridWidth;
    add.s32     %prod, %prod, %x_idx;
    // Byte offset: each float = 4 bytes.
    mul.lo.s32  %prod, %prod, 4;
    cvt.u64.s32 %offset_ptr, %prod;
    add.u64     %density_base, %density_base, %offset_ptr;
    // Atomic add 1.0 to the density grid element.
    atom.global.add.f32 [%density_base], 1.0;

EXIT1:
    ret;
}