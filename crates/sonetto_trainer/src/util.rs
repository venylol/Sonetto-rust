use sonetto_core::board::{Board, Color};
use sonetto_core::zobrist;

/// Parse a side-to-move token into [`Color`].
///
/// Accepted tokens (case-insensitive):
/// - `X` teacher/binary style => Black
/// - `O` teacher/binary style => White
/// - `B`, `BLACK`, `0` => Black
/// - `W`, `WHITE`, `1` => White
pub fn parse_side_token(tok: &str) -> Option<Color> {
    let t = tok.trim();
    if t.is_empty() {
        return None;
    }

    // Common 1-char tokens
    if t.len() == 1 {
        return parse_side_byte(t.as_bytes()[0]);
    }

    // Word tokens
    match t.to_ascii_lowercase().as_str() {
        "black" | "b" => Some(Color::Black),
        "white" | "w" => Some(Color::White),
        _ => None,
    }
}

#[inline(always)]
fn parse_side_byte(b: u8) -> Option<Color> {
    match b {
        b'X' | b'x' | b'B' | b'b' | b'0' => Some(Color::Black),
        b'O' | b'o' | b'W' | b'w' | b'1' => Some(Color::White),
        _ => None,
    }
}

/// Parse a board token.
///
/// Supported input forms:
/// - `64 chars` board only (side defaults to `default_side`).
/// - `65 chars` board+side (side inferred from last char).
///
/// Board encoding:
/// - `X`/`x`/`1`: black
/// - `O`/`o`/`2`: white
/// - `-`/`.`/`0`: empty
pub fn parse_board_token(token: &str, default_side: Color) -> Result<Board, String> {
    let s = token.trim();
    let bytes = s.as_bytes();
    match bytes.len() {
        64 => parse_board64(s, default_side),
        65 => {
            let side = parse_side_byte(bytes[64])
                .ok_or_else(|| format!("invalid side char in board token: {}", bytes[64] as char))?;
            parse_board64(&s[0..64], side)
        }
        n => Err(format!("board token length must be 64 or 65, got {n}")),
    }
}

/// Parse a 64-character board string plus an explicit side.
pub fn parse_board64(board64: &str, side: Color) -> Result<Board, String> {
    let bytes = board64.as_bytes();
    if bytes.len() != 64 {
        return Err(format!("board length must be 64, got {}", bytes.len()));
    }

    let mut black: u64 = 0;
    let mut white: u64 = 0;

    for i in 0..64 {
        match bytes[i] {
            b'X' | b'x' | b'1' => black |= 1u64 << (i as u64),
            b'O' | b'o' | b'2' => white |= 1u64 << (i as u64),
            b'-' | b'.' | b'0' => {}
            other => {
                return Err(format!(
                    "invalid board char at idx {i}: '{}' (byte={other})",
                    other as char
                ));
            }
        }
    }

    let occ = black | white;
    let empty_count = 64u8 - occ.count_ones() as u8;

    let bits_by_color = [black, white];
    let hash = zobrist::compute_hash(bits_by_color, side);

    let (player, opponent) = if side == Color::Black { (black, white) } else { (white, black) };

    Ok(Board {
        player,
        opponent,
        side,
        empty_count,
        hash,
        feat_is_pattern_ids: false,
        feat_id_abs: Vec::new(),
    })
}

/// Convert a board into the canonical compact teacher / protocol form:
/// `64 chars` + `side char`, no whitespace.
pub fn board_to_compact_65(board: &Board) -> String {
    let mut s = String::with_capacity(65);

    let black = board.bits_of(Color::Black);
    let white = board.bits_of(Color::White);

    for i in 0..64 {
        let bit = 1u64 << (i as u64);
        if (black & bit) != 0 {
            s.push('X');
        } else if (white & bit) != 0 {
            s.push('O');
        } else {
            s.push('-');
        }
    }

    s.push(match board.side {
        Color::Black => 'X',
        Color::White => 'O',
    });

    s
}

/// Parse a move coordinate pair like `b4` into internal bitpos (0..63).
///
/// Mapping:
/// - file: `a..h` => col 0..7
/// - rank: `1..8` => row 0..7
pub fn coord_pair_to_bitpos(file: u8, rank: u8) -> Option<u8> {
    let col = match file {
        b'a'..=b'h' => file - b'a',
        b'A'..=b'H' => file - b'A',
        _ => return None,
    };

    let row = match rank {
        b'1'..=b'8' => rank - b'1',
        _ => return None,
    };

    Some(row * 8 + col)
}

/// Convert internal bitpos (0..63) to a coordinate string like `b4`.
pub fn bitpos_to_coord(bitpos: u8) -> String {
    let row = bitpos / 8;
    let col = bitpos % 8;
    let file = (b'a' + col) as char;
    let rank = (b'1' + row) as char;
    format!("{file}{rank}")
}

/// Convert internal bitpos (Sonetto) to the historical Egaroucid-style policy index.
///
/// Egaroucid uses `policy = 63 - (row*8+col)`.
#[inline(always)]
pub const fn teacher_policy_from_bitpos(bitpos: u8) -> u8 {
    63u8.wrapping_sub(bitpos)
}

/// Convert historical teacher policy index back to Sonetto internal bitpos.
#[inline(always)]
pub const fn bitpos_from_teacher_policy(policy: u8) -> u8 {
    63u8.wrapping_sub(policy)
}
