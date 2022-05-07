import {Directive, Input, TemplateRef, ViewContainerRef} from '@angular/core';
import {LogLine} from "./logs-view/logs-view.component";

@Directive({
    selector: '[appLogLines]'
})
export class LogLinesDirective {
    public constructor(private templateRef: TemplateRef<any>,
                       private viewContainer: ViewContainerRef) {
    }
    
    @Input()
    public set appLogLines(lines: LogLine[]) {
        for(let i = this.viewContainer.length; i < lines.length; i++) {
            this.viewContainer.insert(this.templateRef.createEmbeddedView({line: lines[i]}), i);
        }
    }
    
    public static ngTemplateContextGuard(dir: LogLinesDirective, ctx: unknown): ctx is {line: LogLine} {
        return true;
    }
}
