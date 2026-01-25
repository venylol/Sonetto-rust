/*!
 * zip_local.js - minimal ZIP (store) reader/writer for Sonetto offline builds.
 * - No external dependencies.
 * - Writer uses ZIP "store" (no compression) for maximum compatibility.
 * - Reader supports "store" (method 0). For deflate (method 8), it will try
 *   DecompressionStream if available; otherwise throws a clear error.
 */
(function(global){
  'use strict';

  function u16le(v){ return [v & 255, (v>>>8) & 255]; }
  function u32le(v){ return [v & 255, (v>>>8)&255, (v>>>16)&255, (v>>>24)&255]; }

  // CRC32 (IEEE) table
  const CRC_TABLE = (function(){
    const t = new Uint32Array(256);
    for(let i=0;i<256;i++){
      let c=i;
      for(let k=0;k<8;k++){
        c = (c & 1) ? (0xEDB88320 ^ (c>>>1)) : (c>>>1);
      }
      t[i]=c>>>0;
    }
    return t;
  })();

  function crc32(u8){
    let c = 0xFFFFFFFF;
    for(let i=0;i<u8.length;i++){
      c = CRC_TABLE[(c ^ u8[i]) & 255] ^ (c>>>8);
    }
    return (c ^ 0xFFFFFFFF)>>>0;
  }

  const te = new TextEncoder();
  const td = new TextDecoder('utf-8');

  function toU8(data){
    if (data == null) return new Uint8Array(0);
    if (data instanceof Uint8Array) return data;
    if (data instanceof ArrayBuffer) return new Uint8Array(data);
    if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (typeof data === 'string') return te.encode(data);
    throw new Error('ZipLocal: unsupported data type');
  }

  function concatU8(chunks, total){
    const out = new Uint8Array(total);
    let off=0;
    for(const c of chunks){ out.set(c, off); off+=c.length; }
    return out;
  }

  function msdosTimeDate(d){
    // ZIP uses MS-DOS date/time fields. Use local time.
    const dt = d ? new Date(d) : new Date();
    let year = dt.getFullYear();
    if (year < 1980) year = 1980;
    const dosTime = ((dt.getHours() & 31) << 11) | ((dt.getMinutes() & 63) << 5) | ((Math.floor(dt.getSeconds()/2) & 31));
    const dosDate = (((year - 1980) & 127) << 9) | (((dt.getMonth()+1) & 15) << 5) | ((dt.getDate() & 31));
    return {dosTime, dosDate};
  }

  class ZipEntry {
    constructor(name, method, dataU8){
      this.name = name;
      this.method = method|0;
      this._data = dataU8; // uncompressed bytes
    }
    async async(type){
      const t = (type || 'uint8array').toLowerCase();
      if (t === 'string' || t === 'text') return td.decode(this._data);
      if (t === 'arraybuffer' || t === 'arraybufferlike') {
        return this._data.buffer.slice(this._data.byteOffset, this._data.byteOffset + this._data.byteLength);
      }
      if (t === 'uint8array' || t === 'u8') return new Uint8Array(this._data);
      throw new Error('ZipLocal: unsupported async() type: ' + type);
    }
  }

  class ZipArchive {
    constructor(entries){
      this._entries = entries; // Map name -> ZipEntry
    }
    file(name, data){
      // JSZip compatibility:
      // - If data provided: add/replace (only for writers, but allow anyway)
      // - If only name: return ZipEntry
      if (arguments.length >= 2) {
        this._entries.set(name, new ZipEntry(name, 0, toU8(data)));
        return this;
      }
      return this._entries.get(name) || null;
    }
  }

  class ZipWriter extends ZipArchive {
    constructor(){
      super(new Map());
      this._order = [];
    }
    file(name, data){
      if (arguments.length < 2) return super.file(name);
      const u8 = toU8(data);
      if (!this._entries.has(name)) this._order.push(name);
      this._entries.set(name, new ZipEntry(name, 0, u8));
      return this;
    }
    async generateAsync(opts){
      const type = (opts && opts.type) ? String(opts.type).toLowerCase() : 'uint8array';
      // Build local file headers + data
      const locals = [];
      const centrals = [];
      let offset = 0;

      const now = msdosTimeDate();

      for(const name of this._order){
        const ent = this._entries.get(name);
        const data = ent ? ent._data : new Uint8Array(0);
        const nameU8 = te.encode(name);
        const crc = crc32(data);
        const compSize = data.length;
        const uncompSize = data.length;
        const method = 0; // STORE
        const {dosTime, dosDate} = now;

        // Local file header
        const local = [];
        local.push(...u32le(0x04034b50));      // signature
        local.push(...u16le(20));             // version needed
        local.push(...u16le(0));              // flags
        local.push(...u16le(method));         // method
        local.push(...u16le(dosTime));        // mod time
        local.push(...u16le(dosDate));        // mod date
        local.push(...u32le(crc));            // crc32
        local.push(...u32le(compSize));       // comp size
        local.push(...u32le(uncompSize));     // uncomp size
        local.push(...u16le(nameU8.length));  // name len
        local.push(...u16le(0));              // extra len

        const localU8 = new Uint8Array(local);
        locals.push(localU8, nameU8, data);

        // Central directory entry
        const central = [];
        central.push(...u32le(0x02014b50));   // signature
        central.push(...u16le(20));           // version made by
        central.push(...u16le(20));           // version needed
        central.push(...u16le(0));            // flags
        central.push(...u16le(method));       // method
        central.push(...u16le(dosTime));      // time
        central.push(...u16le(dosDate));      // date
        central.push(...u32le(crc));          // crc
        central.push(...u32le(compSize));     // comp
        central.push(...u32le(uncompSize));   // uncomp
        central.push(...u16le(nameU8.length));// name len
        central.push(...u16le(0));            // extra len
        central.push(...u16le(0));            // comment len
        central.push(...u16le(0));            // disk start
        central.push(...u16le(0));            // int attrs
        central.push(...u32le(0));            // ext attrs
        central.push(...u32le(offset));       // local header offset

        const centralU8 = new Uint8Array(central);
        centrals.push(centralU8, nameU8);

        offset += localU8.length + nameU8.length + data.length;
      }

      const centralStart = offset;
      let centralSize = 0;
      for(const c of centrals) centralSize += c.length;
      offset += centralSize;

      // End of central directory
      const eocd = [];
      eocd.push(...u32le(0x06054b50));              // signature
      eocd.push(...u16le(0));                       // disk
      eocd.push(...u16le(0));                       // start disk
      const nFiles = this._order.length;
      eocd.push(...u16le(nFiles));                  // entries on disk
      eocd.push(...u16le(nFiles));                  // total entries
      eocd.push(...u32le(centralSize));             // central size
      eocd.push(...u32le(centralStart));            // central offset
      eocd.push(...u16le(0));                       // comment length
      const eocdU8 = new Uint8Array(eocd);

      const all = [...locals, ...centrals, eocdU8];
      let total = 0;
      for(const c of all) total += c.length;
      const out = concatU8(all, total);

      if (type === 'blob') {
        return new Blob([out], {type:'application/zip'});
      }
      if (type === 'uint8array' || type === 'u8') return out;
      if (type === 'arraybuffer') return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
      throw new Error('ZipLocal: unsupported generateAsync type: ' + type);
    }
  }

  function readU16(u8, o){ return (u8[o] | (u8[o+1]<<8))>>>0; }
  function readU32(u8, o){ return (u8[o] | (u8[o+1]<<8) | (u8[o+2]<<16) | (u8[o+3]<<24))>>>0; }

  async function inflateMaybe(method, compU8){
    if (method === 0) return compU8;
    if (method !== 8) throw new Error('ZipLocal: unsupported compression method: ' + method);
    // Try DecompressionStream with deflate-raw then deflate.
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('ZipLocal: deflate entries require DecompressionStream (modern browser) or a deflate polyfill');
    }
    const tryFmt = async (fmt) => {
      const ds = new DecompressionStream(fmt);
      const inStream = new Blob([compU8]).stream().pipeThrough(ds);
      const ab = await new Response(inStream).arrayBuffer();
      return new Uint8Array(ab);
    };
    try {
      return await tryFmt('deflate-raw');
    } catch (e1) {
      try {
        return await tryFmt('deflate');
      } catch (e2) {
        throw new Error('ZipLocal: cannot inflate deflate entry (need deflate-raw support)');
      }
    }
  }

  function findEOCD(u8){
    // EOCD signature 0x06054b50; search backwards up to 66k
    const min = Math.max(0, u8.length - 65557);
    for(let i=u8.length-22; i>=min; i--){
      if (u8[i]===0x50 && u8[i+1]===0x4b && u8[i+2]===0x05 && u8[i+3]===0x06) return i;
    }
    return -1;
  }

  async function loadZipAsync(blobOrBuf){
    const u8 = toU8(blobOrBuf instanceof Blob ? await blobOrBuf.arrayBuffer() : blobOrBuf);
    const eocdOff = findEOCD(u8);
    if (eocdOff < 0) throw new Error('ZipLocal: invalid zip (EOCD not found)');
    const totalEntries = readU16(u8, eocdOff + 10);
    const centralSize = readU32(u8, eocdOff + 12);
    const centralOff  = readU32(u8, eocdOff + 16);

    const entries = new Map();
    let p = centralOff;
    for(let i=0; i<totalEntries; i++){
      if (readU32(u8, p) !== 0x02014b50) throw new Error('ZipLocal: invalid central directory');
      const method = readU16(u8, p + 10);
      const crc = readU32(u8, p + 16);
      const compSize = readU32(u8, p + 20);
      const uncompSize = readU32(u8, p + 24);
      const nameLen = readU16(u8, p + 28);
      const extraLen = readU16(u8, p + 30);
      const commentLen = readU16(u8, p + 32);
      const lho = readU32(u8, p + 42);
      const name = td.decode(u8.slice(p + 46, p + 46 + nameLen));
      p = p + 46 + nameLen + extraLen + commentLen;

      // Read local file header to locate data
      if (readU32(u8, lho) !== 0x04034b50) throw new Error('ZipLocal: invalid local header');
      const lfNameLen = readU16(u8, lho + 26);
      const lfExtraLen = readU16(u8, lho + 28);
      const dataOff = lho + 30 + lfNameLen + lfExtraLen;
      const compU8 = u8.slice(dataOff, dataOff + compSize);
      const dataU8 = await inflateMaybe(method, compU8);

      entries.set(name, new ZipEntry(name, method, dataU8));
    }
    return new ZipArchive(entries);
  }

  const ZipLocal = {
    create: () => new ZipWriter(),
    loadAsync: loadZipAsync,
  };

  global.ZipLocal = ZipLocal;
})(typeof window !== 'undefined' ? window : self);
