import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root=path.resolve(process.argv[2]);
const {parseArgs}=await import(pathToFileURL(path.join(root,'tools/performance_suite.mjs')).href);
const cases=[];
const flags=[['--runs','runs'],['--warmup','warmupRuns'],['--timeout-ms','timeoutMs'],['--rdna-vk-device','rdnaVkDevice']];
function check(label,fn){try{fn();cases.push({label,passed:true});}catch(e){cases.push({label,passed:false,error:e.message});}}
for(const [flag,key] of flags){
 for(const value of ['1','12','0003','9007199254740991'])
  check(flag+' valid '+value,()=>assert.equal(parseArgs([flag,value])[key],Number(value)));
 for(const value of ['3junk','3.5','1e3','0x10','+3',' 3','3 ','Infinity','NaN','-1','','9007199254740992','9007199254740993','999999999999999999999999999999999999'])
  check(flag+' invalid '+JSON.stringify(value),()=>assert.throws(()=>parseArgs([flag,value])));
 check(flag+' missing',()=>assert.throws(()=>parseArgs([flag])));
}
check('zero runs rejected',()=>assert.throws(()=>parseArgs(['--runs','0'])));
for(const [flag,key] of flags.slice(1))
 check(flag+' zero allowed',()=>assert.equal(parseArgs([flag,'0'])[key],0));
console.log(JSON.stringify({task:'strict-numbers',passed:cases.every(c=>c.passed),cases},null,2));
if(cases.some(c=>!c.passed))process.exitCode=1;
