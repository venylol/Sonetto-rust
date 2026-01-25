/*
  Egbk3Book (Egaroucid .egbk3 opening book)

  Notes / constraints:
  - This is a clean-room re-implementation based on the public on-disk format
    and documented symmetry rules. No code is copied from Egaroucid.
  - Designed for modern browsers (BigInt supported). No runtime compilation.
*/

(function (global) {
  'use strict';

  // --- Constants (matching Egaroucid book v3 file format) ---
  const EGBK3_MAGIC = 'DICUORAGE'; // 9 bytes
  const EGBK3_VERSION = 3;
  const HEADER_SIZE = 9 + 1 + 4; // magic + version(u8) + n_boards(u32)
  const RECORD_SIZE = 25; // per-board record

  // Egaroucid engine constants (for leaf_move sentinels)
  const SCORE_UNDEFINED = -126;
  const MOVE_PASS = 64;
  const MOVE_NOMOVE = 65;
  const MOVE_UNDEFINED = 125;

  // Book random parameters
  const BOOK_ACCURACY_LEVEL_INF = 10;
  const BOOK_LOSS_IGNORE_THRESHOLD = 8;

  // --- Utilities ---
  function readAscii(view, off, len) {
    let s = '';
    for (let i = 0; i < len; i++) {
      s += String.fromCharCode(view.getUint8(off + i));
    }
    return s;
  }

  function readU64LE(view, off) {
    // Avoid DataView.getBigUint64 for compatibility (still uses BigInt).
    const lo = view.getUint32(off, true);
    const hi = view.getUint32(off + 4, true);
    return (BigInt(hi) << 32n) | BigInt(lo);
  }

  function clampInt(n, lo, hi) {
    n = n | 0;
    if (n < lo) return lo;
    if (n > hi) return hi;
    return n;
  }

  function pickBestDeterministic(candidates) {
    // Stable tie-break: higher val first, then smaller (y,x).
    // This keeps analysis deterministic and makes acc_level=0 truly "no randomness".
    let best = null;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      if (!best) {
        best = c;
        continue;
      }
      if (c.val > best.val) {
        best = c;
        continue;
      }
      if (c.val === best.val) {
        if ((c.y | 0) < (best.y | 0) || ((c.y | 0) === (best.y | 0) && (c.x | 0) < (best.x | 0))) {
          best = c;
        }
      }
    }
    return best;
  }

  function nextPow2(n) {
    // n: positive integer
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  // Precompute bit masks for cells 0..63 (Egaroucid cell index == bit index)
  const BIT = new Array(64);
  for (let i = 0; i < 64; i++) BIT[i] = 1n << BigInt(i);

  // --- Coordinate mapping between Sonetto (x,y) and Egaroucid cell index ---
  // Sonetto UI uses (x=0..7, y=0..7) where (0,0) is top-left == "a1".
  // Egaroucid uses cell index with: idx = 63 - (y*8 + x).
  function sonettoXYToEgarCell(x, y) {
    return 63 - (y * 8 + x);
  }

  function egarCellToSonettoXY(cell) {
    // Inverse of idx = 63 - (y*8 + x)
    const inv = 63 - cell;
    const y = (inv / 8) | 0;
    const x = (inv % 8) | 0;
    return { x, y };
  }

  // --- Symmetry coordinate transforms (idx 0..7) ---
  // Definitions follow the same formulas as the Egaroucid symmetry utilities.
  function coordToRepresentative(cell, idx) {
    const y = (cell / 8) | 0;
    const x = (cell % 8) | 0;
    switch (idx) {
      case 0: return cell;
      case 1: return (7 - y) * 8 + x;              // vertical
      case 2: return (7 - x) * 8 + (7 - y);        // black line
      case 3: return x * 8 + (7 - y);              // black line + vertical (rotate 90 cw)
      case 4: return (7 - x) * 8 + y;              // black line + horizontal (rotate 90 ccw)
      case 5: return x * 8 + y;                    // white line
      case 6: return y * 8 + (7 - x);              // horizontal
      case 7: return (7 - y) * 8 + (7 - x);        // horizontal + vertical (rotate 180)
      default: return -1;
    }
  }

  function coordFromRepresentative(cell, idx) {
    const y = (cell / 8) | 0;
    const x = (cell % 8) | 0;
    switch (idx) {
      case 0: return cell;
      case 1: return (7 - y) * 8 + x;              // vertical
      case 2: return (7 - x) * 8 + (7 - y);        // black line
      case 3: return (7 - x) * 8 + y;              // black line + vertical (rotate 90 cw)
      case 4: return x * 8 + (7 - y);              // black line + horizontal (rotate 90 ccw)
      case 5: return x * 8 + y;                    // white line
      case 6: return y * 8 + (7 - x);              // horizontal
      case 7: return (7 - y) * 8 + (7 - x);        // horizontal + vertical (rotate 180)
      default: return -1;
    }
  }

  // Precompute mapping tables for bitboard transforms.
  const TO_REP = Array.from({ length: 8 }, () => new Int8Array(64));
  const FROM_REP = Array.from({ length: 8 }, () => new Int8Array(64));
  for (let idx = 0; idx < 8; idx++) {
    for (let c = 0; c < 64; c++) {
      TO_REP[idx][c] = coordToRepresentative(c, idx);
      FROM_REP[idx][c] = coordFromRepresentative(c, idx);
    }
  }

  function transformBitboard(bb, idx) {
    if (idx === 0) return bb;
    let out = 0n;
    const map = TO_REP[idx];
    // Scan all 64 cells (fast enough for our call frequency).
    for (let c = 0; c < 64; c++) {
      if (bb & BIT[c]) {
        out |= BIT[map[c]];
      }
    }
    return out;
  }

  function transformPair(player, opponent, idx) {
    return {
      player: transformBitboard(player, idx),
      opponent: transformBitboard(opponent, idx)
    };
  }

  function isSmallerPair(pA, oA, pB, oB) {
    // Lexicographic compare by (player, opponent) choosing the smallest.
    return (pB < pA) || (pB === pA && oB < oA);
  }

  function representativePair(player, opponent) {
    // Same scan order as Egaroucid representative_board(b, &idx):
    // start idx=0, then: 2,1,3,6,4,7,5
    let bestP = player;
    let bestO = opponent;
    let bestIdx = 0;

    const order = [2, 1, 3, 6, 4, 7, 5];
    for (let k = 0; k < order.length; k++) {
      const idx = order[k];
      const t = transformPair(player, opponent, idx);
      if (isSmallerPair(bestP, bestO, t.player, t.opponent)) {
        bestP = t.player;
        bestO = t.opponent;
        bestIdx = idx;
      }
    }
    return { player: bestP, opponent: bestO, idx: bestIdx };
  }

  // --- Bitboard movegen / apply (probe hot path) ---
  //
  // Performance goal:
  //   During book probing we repeatedly do:
  //     enumerate legal moves -> apply(move) -> (maybe pass) -> lookup(child)
  //   The previous implementation used board-array scanning, allocating flip lists
  //   and copying the whole 8x8 board for each candidate. That quickly becomes the
  //   dominant cost.
  //
  // New approach:
  //   Use 64-bit BigInt bitboards and fixed shift/mask propagation to compute flips
  //   and "opponent has any legal move" (pass check) without scanning the board.
  //   This is only used in the book probe path; the rest of the UI can keep using
  //   its existing board representation.
  //
  // Representation:
  //   bit i (0..63) corresponds to Egaroucid cell index i, consistent with
  //   boardToEgarBitboards() and symmetry utilities in this file.

  const BB_FULL = 0xFFFFFFFFFFFFFFFFn;
  const BB_FILE_A = 0x0101010101010101n;
  const BB_FILE_H = 0x8080808080808080n;
  const BB_NOT_A = BB_FULL ^ BB_FILE_A;
  const BB_NOT_H = BB_FULL ^ BB_FILE_H;

  function bbE(x) { return ((x & BB_NOT_H) << 1n) & BB_FULL; }
  function bbW(x) { return (x & BB_NOT_A) >> 1n; }
  function bbS(x) { return (x << 8n) & BB_FULL; }
  function bbN(x) { return x >> 8n; }
  function bbSE(x) { return ((x & BB_NOT_H) << 9n) & BB_FULL; }
  function bbSW(x) { return ((x & BB_NOT_A) << 7n) & BB_FULL; }
  function bbNE(x) { return (x & BB_NOT_H) >> 7n; }
  function bbNW(x) { return (x & BB_NOT_A) >> 9n; }

  function bbDirMoves(player, opponent, empty, shiftFn) {
    // Fixed 6-step expansion (max 6 opponent discs can be sandwiched on 8x8).
    let t = shiftFn(player) & opponent;
    for (let i = 0; i < 5; i++) t |= shiftFn(t) & opponent;
    return shiftFn(t) & empty;
  }

  function hasAnyLegalMoveBB(player, opponent) {
    const empty = (~(player | opponent)) & BB_FULL;
    if (empty === 0n) return false;

    // Early exit per direction.
    if (bbDirMoves(player, opponent, empty, bbE) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbW) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbS) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbN) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbSE) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbSW) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbNE) !== 0n) return true;
    if (bbDirMoves(player, opponent, empty, bbNW) !== 0n) return true;

    return false;
  }

  function bbDirFlips(moveBit, player, opponent, shiftFn) {
    let t = shiftFn(moveBit) & opponent;
    for (let i = 0; i < 5; i++) t |= shiftFn(t) & opponent;
    // Bracketed by player's disc?
    return (shiftFn(t) & player) !== 0n ? t : 0n;
  }

  function applyMoveWithOptionalPassBB(player, opponent, moveBit) {
    // Returns a child position (bitboards are always from side-to-move perspective)
    // plus sgn that converts child.value to current player's perspective.
    // - If opponent can move: next is swapped, sgn=-1
    // - If opponent must pass: next is same player, sgn=+1

    if ((moveBit & (player | opponent)) !== 0n) return null; // occupied

    let flips = 0n;
    flips |= bbDirFlips(moveBit, player, opponent, bbE);
    flips |= bbDirFlips(moveBit, player, opponent, bbW);
    flips |= bbDirFlips(moveBit, player, opponent, bbS);
    flips |= bbDirFlips(moveBit, player, opponent, bbN);
    flips |= bbDirFlips(moveBit, player, opponent, bbSE);
    flips |= bbDirFlips(moveBit, player, opponent, bbSW);
    flips |= bbDirFlips(moveBit, player, opponent, bbNE);
    flips |= bbDirFlips(moveBit, player, opponent, bbNW);

    if (flips === 0n) return null; // illegal

    const player2 = player | moveBit | flips;
    const opp2 = opponent & ~flips;

    if (hasAnyLegalMoveBB(opp2, player2)) {
      // Opponent plays next.
      return { nextPlayer: opp2, nextOpponent: player2, sgn: -1 };
    }
    // Pass: same player moves again.
    return { nextPlayer: player2, nextOpponent: opp2, sgn: 1 };
  }

  function boardToEgarBitboards(board, playerToMove) {
    let black = 0n;
    let white = 0n;
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const v = board[x][y] | 0;
        if (v !== 1 && v !== 2) continue;
        const cell = sonettoXYToEgarCell(x, y);
        if (v === 1) black |= BIT[cell];
        else white |= BIT[cell];
      }
    }
    if ((black & white) !== 0n) {
      // Illegal board; return something deterministic.
      // (We keep it non-throwing to avoid breaking UI.)
      white &= ~black;
    }
    const player = (playerToMove === 1) ? black : white;
    const opponent = (playerToMove === 1) ? white : black;
    return { player, opponent };
  }

  // --- Egbk3Book: parse + index + query ---
  class Egbk3Book {
    constructor(arrayBuffer) {
      if (!(arrayBuffer instanceof ArrayBuffer)) {
        throw new Error('Egbk3Book expects an ArrayBuffer');
      }
      this.buffer = arrayBuffer;
      this.view = new DataView(arrayBuffer);
      this.nBoards = 0;
      this.tableSize = 0;
      this.keyPlayer = null;     // BigUint64Array
      this.keyOpponent = null;   // BigUint64Array
      this.recIndex = null;      // Uint32Array
      this.ready = false;

      // Probe cache (hot path): key is a 128-bit BigInt composed from
      // (playerBB << 64) | opponentBB, both 64-bit.
      //
      // Why cache here?
      // - Analysis UI and AI can query the same position repeatedly (undo/redo,
      //   toggling analysis, re-render, etc.).
      // - Probing requires N child lookups + symmetry canonicalization per child.
      //
      // Cache is bounded to avoid unbounded memory growth.
      this._probeCache = new Map();
      this._probeCacheMax = 4096;

      // Temporary weights buffer to avoid per-call allocations in getRandomMove.
      this._tmpWeights = new Float64Array(64);

      this._parseHeader();
    }

    _parseHeader() {
      if (this.buffer.byteLength < HEADER_SIZE) {
        throw new Error('egbk3: file too small');
      }
      const magic = readAscii(this.view, 0, 9);
      if (magic !== EGBK3_MAGIC) {
        throw new Error(`egbk3: bad magic '${magic}'`);
      }
      const version = this.view.getUint8(9);
      if (version !== EGBK3_VERSION) {
        throw new Error(`egbk3: unsupported version ${version}`);
      }
      this.nBoards = this.view.getUint32(10, true);

      const expectedMinSize = HEADER_SIZE + this.nBoards * RECORD_SIZE;
      if (this.buffer.byteLength < expectedMinSize) {
        throw new Error(`egbk3: truncated file (need >= ${expectedMinSize}, got ${this.buffer.byteLength})`);
      }
    }

    _hashSlot(p, o, maskBig) {
      // Simple 64-bit mix (splitmix-inspired). Clean-room.
      let x = p ^ (o * 0x9e3779b97f4a7c15n);
      x ^= x >> 32n;
      x *= 0xbf58476d1ce4e5b9n;
      x ^= x >> 29n;
      return Number(x & maskBig);
    }

    async buildIndex(opts) {
      if (this.ready) return;
      const onProgress = opts && typeof opts.onProgress === 'function' ? opts.onProgress : null;

      if (typeof BigUint64Array === 'undefined') {
        throw new Error('egbk3: BigUint64Array not supported in this browser');
      }

      // Keep load factor <= ~0.70.
      const target = Math.ceil(this.nBoards / 0.70);
      const size = nextPow2(Math.max(1024, target));
      this.tableSize = size;

      this.keyPlayer = new BigUint64Array(size);
      this.keyOpponent = new BigUint64Array(size);
      this.recIndex = new Uint32Array(size);
      this.recIndex.fill(0xFFFFFFFF);

      const mask = size - 1;
      const maskBig = BigInt(mask);

      const reportEvery = 1 << 14; // 16384
      for (let i = 0; i < this.nBoards; i++) {
        const off = HEADER_SIZE + i * RECORD_SIZE;
        const p = readU64LE(this.view, off);
        const o = readU64LE(this.view, off + 8);
        // Skip obviously illegal overlaps (defensive).
        if ((p & o) !== 0n) continue;

        let slot = this._hashSlot(p, o, maskBig);
        while (true) {
          const ri = this.recIndex[slot];
          if (ri === 0xFFFFFFFF) {
            this.keyPlayer[slot] = p;
            this.keyOpponent[slot] = o;
            this.recIndex[slot] = i >>> 0;
            break;
          }
          if (this.keyPlayer[slot] === p && this.keyOpponent[slot] === o) {
            // Duplicate key in file: keep the first one (stable).
            break;
          }
          slot = (slot + 1) & mask;
        }

        if (onProgress && (i % reportEvery) === 0) {
          onProgress(i / this.nBoards);
          // Yield to UI occasionally.
          // eslint-disable-next-line no-await-in-loop
          await new Promise(r => setTimeout(r, 0));
        }
      }

      this.ready = true;
      if (onProgress) onProgress(1);
    }

    _findRecordIndex(repPlayer, repOpponent) {
      if (!this.ready) return -1;
      const size = this.tableSize;
      const mask = size - 1;
      const maskBig = BigInt(mask);

      let slot = this._hashSlot(repPlayer, repOpponent, maskBig);
      while (true) {
        const ri = this.recIndex[slot];
        if (ri === 0xFFFFFFFF) return -1;
        if (this.keyPlayer[slot] === repPlayer && this.keyOpponent[slot] === repOpponent) {
          return ri | 0;
        }
        slot = (slot + 1) & mask;
      }
    }

    _readRecordByIndex(recIdx, repIdxForLeafRestore) {
      // repIdxForLeafRestore: representative idx used for this query.
      const off = HEADER_SIZE + recIdx * RECORD_SIZE;
      // Key fields are not strictly needed (we already matched them),
      // but reading is cheap and helps debugging.
      const player = readU64LE(this.view, off);
      const opponent = readU64LE(this.view, off + 8);
      const value = this.view.getInt8(off + 16);
      const level = this.view.getInt8(off + 17);
      const nLines = this.view.getUint32(off + 18, true);
      const leafValue = this.view.getInt8(off + 22);
      const leafMoveRaw = this.view.getInt8(off + 23);
      const leafLevel = this.view.getInt8(off + 24);

      let leafMoveCell = leafMoveRaw;
      let leafMoveXY = null;
      if (leafMoveRaw >= 0 && leafMoveRaw < 64) {
        // Convert from representative orientation back to the original board orientation.
        const restoredCell = FROM_REP[repIdxForLeafRestore][leafMoveRaw];
        leafMoveCell = restoredCell;
        leafMoveXY = egarCellToSonettoXY(restoredCell);
      }

      return {
        keyPlayer: player,
        keyOpponent: opponent,
        value,
        level,
        nLines,
        leaf: {
          value: leafValue,
          moveRaw: leafMoveRaw,
          moveCell: leafMoveCell,
          moveXY: leafMoveXY,
          level: leafLevel
        }
      };
    }

    lookupByBitboards(playerBB, opponentBB) {
      const rep = representativePair(playerBB, opponentBB);
      const recIdx = this._findRecordIndex(rep.player, rep.opponent);
      if (recIdx < 0) return null;
      const r = this._readRecordByIndex(recIdx, rep.idx);
      return {
        value: r.value,
        level: r.level,
        nLines: r.nLines,
        leaf: r.leaf,
        repIdx: rep.idx,
        repPlayer: rep.player,
        repOpponent: rep.opponent
      };
    }

    lookup(board, playerToMove) {
      if (!this.ready) return null;
      const bb = boardToEgarBitboards(board, playerToMove);
      return this.lookupByBitboards(bb.player, bb.opponent);
    }

    collectCandidates(board, playerToMove, validMoves) {
      // Deterministic probe: collect move values where the resulting position is present in book.
      //
      // Hot path: used by both
      // - analysis display (deterministic list)
      // - AI book move (random policy selection)
      //
      // Performance: we convert the board to bitboards once, then for each move we apply it
      // via bitboard flips (no board copy, no flip list allocations) and check "pass" via
      // a boolean legal-move existence test.
      const moves = Array.isArray(validMoves) ? validMoves : [];
      if (!this.ready) {
        return { candidates: [], bestScore: -Infinity, selfValue: SCORE_UNDEFINED, reliable: false };
      }

      const bb = boardToEgarBitboards(board, playerToMove);
      const playerBB = bb.player;
      const opponentBB = bb.opponent;

      // Cache key is unique for (side-to-move playerBB, opponentBB).
      const key128 = (playerBB << 64n) | opponentBB;

      // movesMask is a defensive cache guard: callers *should* always pass all legal moves,
      // but keeping this costs little and avoids incorrect reuse if someone calls with a
      // filtered move list.
      let movesMask = 0n;
      for (let i = 0; i < moves.length; i++) {
        const m = moves[i];
        const x = m.x | 0;
        const y = m.y | 0;
        if ((x | y) & ~7) continue;
        const cell = sonettoXYToEgarCell(x, y);
        movesMask |= BIT[cell];
      }

      const cached = this._probeCache.get(key128);
      if (cached && cached.movesMask === movesMask) {
        return cached.probe;
      }

      const candidates = [];
      let bestScore = -Infinity;

      for (let i = 0; i < moves.length; i++) {
        const m = moves[i];
        const x = m.x | 0;
        const y = m.y | 0;
        if ((x | y) & ~7) continue;

        const cell = sonettoXYToEgarCell(x, y);
        const moveBit = BIT[cell];

        const applied = applyMoveWithOptionalPassBB(playerBB, opponentBB, moveBit);
        if (!applied) continue;

        const childEntry = this.lookupByBitboards(applied.nextPlayer, applied.nextOpponent);
        if (!childEntry) continue;

        const v = (applied.sgn | 0) * (childEntry.value | 0);
        if (v > bestScore) bestScore = v;
        candidates.push({ x, y, val: v, isBookHit: true });
      }

      const selfEntry = this.lookupByBitboards(playerBB, opponentBB);
      const selfValue = selfEntry ? (selfEntry.value | 0) : SCORE_UNDEFINED;

      const reliable = (candidates.length > 0) && (bestScore >= (selfValue - BOOK_LOSS_IGNORE_THRESHOLD));

      const probe = {
        candidates,
        bestScore: candidates.length > 0 ? bestScore : -Infinity,
        selfValue,
        reliable
      };

      // Simple bounded LRU (insertion order): delete oldest when exceeding max.
      this._probeCache.set(key128, { movesMask, probe });
      if (this._probeCache.size > this._probeCacheMax) {
        const oldest = this._probeCache.keys().next().value;
        this._probeCache.delete(oldest);
      }

      return probe;
    }

    getMoveValues(board, playerToMove, validMoves) {
      const probe = this.collectCandidates(board, playerToMove, validMoves);
      if (!probe.reliable) return null;
      // Sort best-first for analysis display.
      const out = probe.candidates.slice();
      out.sort((a, b) => (b.val - a.val) || ((a.y - b.y) || (a.x - b.x)));
      return out;
    }

    getRandomMove(board, playerToMove, validMoves, accLevel) {
      // Implements Egaroucid-like get_random(acc_level) policy selection.
      const acc = clampInt(parseInt(accLevel, 10) || 0, 0, 10);
      const probe = this.collectCandidates(board, playerToMove, validMoves);
      if (!probe.reliable) return null;

      const candidates = probe.candidates;
      if (candidates.length === 0) return null;
      const bestScore = probe.bestScore;

      // acc_level=0: explicitly deterministic (no randomness).
      // This is a user-facing UX requirement for "0 means no random".
      if (acc === 0) {
        const best = pickBestDeterministic(candidates);
        return best ? { x: best.x, y: best.y, val: best.val, isBookHit: true } : null;
      }

      // Filter by acceptable_min_value and compute weights.
      const acceptableMin = bestScore - 2.0 * acc - 0.5;
      const powExp = BOOK_ACCURACY_LEVEL_INF - acc;
      let sum = 0.0;

      // Reuse typed buffer to avoid allocations.
      if (!this._tmpWeights || this._tmpWeights.length < candidates.length) {
        this._tmpWeights = new Float64Array(Math.max(64, candidates.length));
      }
      const wbuf = this._tmpWeights;

      for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i];
        let w = 0.0;
        if (c.val >= acceptableMin) {
          const expVal = (Math.exp(c.val - bestScore) + 1.5) / 3.0;
          w = Math.pow(expVal, powExp);
        }
        wbuf[i] = w;
        sum += w;
      }

      if (!(sum > 0)) {
        // Defensive fallback: choose best deterministically.
        const best = pickBestDeterministic(candidates);
        return best ? { x: best.x, y: best.y, val: best.val, isBookHit: true } : null;
      }

      // Weighted random sampling.
      let rnd = Math.random() * sum;
      let accW = 0.0;
      for (let i = 0; i < candidates.length; i++) {
        const w = wbuf[i];
        if (!(w > 0)) continue;
        accW += w;
        if (accW >= rnd) {
          const c = candidates[i];
          return { x: c.x, y: c.y, val: c.val, isBookHit: true };
        }
      }

      // Numerical drift fallback: choose the last positive weight.
      for (let i = candidates.length - 1; i >= 0; i--) {
        if (wbuf[i] > 0) {
          const c = candidates[i];
          return { x: c.x, y: c.y, val: c.val, isBookHit: true };
        }
      }
      const best = pickBestDeterministic(candidates);
      return best ? { x: best.x, y: best.y, val: best.val, isBookHit: true } : null;
    }

    // Convenience loader
    static async loadFromUrl(url, opts) {
      const res = await fetch(url, { cache: 'force-cache' });
      if (!res.ok) {
        throw new Error(`egbk3: failed to fetch ${url} (${res.status})`);
      }
      const buf = await res.arrayBuffer();
      const book = new Egbk3Book(buf);
      await book.buildIndex(opts);
      return book;
    }
  }

  // Expose to global
  global.Egbk3Book = Egbk3Book;
  global.__EGBK3_BOOK_CONSTANTS__ = {
    SCORE_UNDEFINED,
    MOVE_PASS,
    MOVE_NOMOVE,
    MOVE_UNDEFINED
  };
})(typeof window !== 'undefined' ? window : globalThis);
