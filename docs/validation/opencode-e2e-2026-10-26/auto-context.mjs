import assert from 'node:assert/strict';
import {mkdtemp,readFile,readdir,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
const root=path.resolve(process.argv[2]);
const helper=path.join(root,'tools/setup_opencode.mjs');
const sandbox=await mkdtemp(path.join(tmpdir(),'zinc-auto-accept-'));
let reply={data:[{id:'qwen38-27b-q4k-m',context_length:262144}]},status=200,hang=false;
const requests=[];
const server=Bun.serve({hostname:'127.0.0.1',port:0,async fetch(req){
 requests.push({pathname:new URL(req.url).pathname,authorization:req.headers.get('authorization')});
 if(hang)return await new Promise(()=>{});
 return new Response(typeof reply==='string'?reply:JSON.stringify(reply),{status,headers:{'content-type':'application/json'}});
}});
const endpoint='http://127.0.0.1:'+server.port+'/v1';
const cases=[];
async function cli(args,extraEnv={}){
 const child=Bun.spawn([process.execPath,helper,...args],{cwd:root,env:{...process.env,ZINC_API_BASE_URL:endpoint,ZINC_API_KEY:'acceptance-fake-key',...extraEnv},stdout:'pipe',stderr:'pipe'});
 const timer=setTimeout(()=>child.kill('SIGKILL'),14500);
 try{const [out,err,code]=await Promise.all([new Response(child.stdout).text(),new Response(child.stderr).text(),child.exited]);return {out,err,code};}
 finally{clearTimeout(timer);}
}
async function check(label,fn){try{await fn();cases.push({label,passed:true});}catch(e){cases.push({label,passed:false,error:e.message});}}
function configOf(res,context){assert.equal(res.code,0,res.err);const config=JSON.parse(res.out);assert.equal(config.model,'zinc/qwen38-27b-q4k-m');assert.equal(config.provider.zinc.models['qwen38-27b-q4k-m'].limit.context,context);assert.equal(config.provider.zinc.options.baseURL,'http://127.0.0.1:9090/v1');assert.equal(config.provider.zinc.options.apiKey,'{env:ZINC_API_KEY}');assert(!res.out.includes('acceptance-fake-key'));assert(!res.out.includes(endpoint));return config;}
async function invalid(label,data,httpStatus=200){
 await check(label,async()=>{reply=data;status=httpStatus;const project=path.join(sandbox,'missing-'+cases.length);const r=await cli(['--project',project,'--context','auto']);assert.notEqual(r.code,0);assert(!r.err.includes('acceptance-fake-key'));assert(!r.err.includes(endpoint));await assert.rejects(()=>readdir(project));});status=200;
}
try{
 for(const context of [8192,32768,262144])await check('discover '+context,async()=>{
  reply={data:[{id:'qwen38-27b-q4k-m',context_length:context}]};const at=requests.length;
  const r=await cli(['--dry-run','--context','auto']);const c=configOf(r,context);
  assert.equal(c.provider.zinc.models['qwen38-27b-q4k-m'].limit.output,Math.min(4096,Math.floor(context/4)));
  assert.deepEqual(requests.slice(at),[{pathname:'/v1/models',authorization:'Bearer acceptance-fake-key'}]);
 });
 await check('exact match in multi-model discovery',async()=>{
  reply={data:[{id:'other',context_length:8192},{id:'qwen38-27b-q4k-m',context_length:65536}]};
  configOf(await cli(['setup','--dry-run','--context','auto']),65536);
 });
 await check('single loaded model permits client alias',async()=>{reply={data:[{id:'loaded.gguf',context_length:32768}]};configOf(await cli(['--dry-run','--context','auto']),32768);});
 await check('writes portable config and hook',async()=>{
  reply={data:[{id:'loaded.gguf',context_length:262144}]};
  const project=path.join(sandbox,'project');
  const r=await cli(['--project',project,'--context','auto']);assert.equal(r.code,0,r.err);
  const text=await readFile(path.join(project,'opencode.zinc.json'),'utf8');
  const c=configOf({out:text,err:'',code:0},262144);
  assert.equal(c.plugin.length,1);assert((await readFile(path.join(project,c.plugin[0]),'utf8')).includes('tool.execute.before'));
  assert(!text.includes(sandbox));assert(!text.includes(endpoint));
 });
 await check('dry-run preserves every existing file',async()=>{
  reply={data:[{id:'loaded.gguf',context_length:32768}]};
  const project=path.join(sandbox,'project');const old=await readFile(path.join(project,'opencode.zinc.json'),'utf8');
  const before=await readdir(project);
  configOf(await cli(['--project',project,'--context','auto','--dry-run']),32768);
  assert.equal(await readFile(path.join(project,'opencode.zinc.json'),'utf8'),old);assert.deepEqual(await readdir(project),before);
 });
 await check('explicit numeric context remains offline',async()=>{
  const at=requests.length;configOf(await cli(['--dry-run','--context','32768'],{ZINC_API_BASE_URL:'http://do-not-contact.invalid/v1'}),32768);assert.equal(requests.length,at);
 });
 await check('scope is setup-only before project/task IO',async()=>{
  for(const command of ['check','run']){const r=await cli([command,'--project',path.join(sandbox,'absent'),'--context','auto']);assert.notEqual(r.code,0);assert.match(r.err,/setup/i);}
 });
 await check('help and guide document automatic discovery',async()=>{
  const r=await cli(['--help']);assert.equal(r.code,0);assert.match(r.out,/--context[^\n]*auto/i);
  const docs=await readFile(path.join(root,'docs/OPENCODE.md'),'utf8');assert.match(docs,/--context auto/);assert.match(docs,/discover|\/v1\/models/i);
 });
 await check('lookup uses output and steps overrides',async()=>{
  reply={data:[{id:'loaded.gguf',context_length:65536}]};
  const c=configOf(await cli(['--dry-run','--context','auto','--output','1024','--steps','12']),65536);
  assert.equal(c.provider.zinc.models['qwen38-27b-q4k-m'].limit.output,1024);assert.equal(c.agent.zinc.steps,12);
 });
 await invalid('ambiguous aliases rejected',{data:[{id:'a',context_length:32768},{id:'b',context_length:65536}]});
 await invalid('missing context rejected',{data:[{id:'loaded.gguf'}]});
 await invalid('string context rejected',{data:[{id:'loaded.gguf',context_length:'32768'}]});
 await invalid('fraction context rejected',{data:[{id:'loaded.gguf',context_length:32768.5}]});
 await invalid('small context rejected',{data:[{id:'loaded.gguf',context_length:4096}]});
 await invalid('unsafe context rejected',{data:[{id:'loaded.gguf',context_length:9007199254740992}]});
 await invalid('empty discovery rejected',{data:[]});
 await invalid('HTTP authentication error fails closed',{message:'do not echo acceptance-fake-key'},401);
 await invalid('malformed JSON fails closed','{"data":[}');
 await check('bounded discovery timeout',async()=>{
  hang=true;const t=performance.now();const r=await cli(['--dry-run','--context','auto']);assert.notEqual(r.code,0);assert(performance.now()-t<12500,'lookup exceeded 12.5 seconds');assert(!r.err.includes('acceptance-fake-key'));hang=false;
 });
 if(process.env.ZINC_E2E_LIVE_BASE)await check('actual ZINC model discovery',async()=>{
  const r=await cli(['--dry-run','--context','auto'],{ZINC_API_BASE_URL:process.env.ZINC_E2E_LIVE_BASE,ZINC_API_KEY:'local'});
  configOf(r,Number(process.env.ZINC_E2E_LIVE_CONTEXT));
 });
 console.log(JSON.stringify({task:'auto-context',passed:cases.every(c=>c.passed),cases},null,2));
 if(cases.some(c=>!c.passed))process.exitCode=1;
}finally{server.stop(true);await rm(sandbox,{recursive:true,force:true});}
