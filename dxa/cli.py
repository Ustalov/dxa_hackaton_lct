import argparse,json,tempfile
from pathlib import Path
from .service import Analyzer,safe_extract

def main():
    p=argparse.ArgumentParser(description='Local DXA quality analysis')
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--artifacts',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts')
    p.add_argument('--variant',default=None)
    args=p.parse_args();analyzer=Analyzer(args.artifacts)
    if args.input.is_dir():
        files=sorted(f for f in args.input.rglob('*') if f.is_file())
        if not files:raise ValueError('Input directory is empty')
        result=analyzer.analyze(files,args.output,base=args.input,variant=args.variant)
    elif args.input.suffix.lower()=='.zip':
        with tempfile.TemporaryDirectory(prefix='dxa-') as temp:
            files=safe_extract(args.input,temp);result=analyzer.analyze(files,args.output,base=Path(temp),variant=args.variant)
    else:result=analyzer.analyze([args.input],args.output,base=args.input.parent,variant=args.variant)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
