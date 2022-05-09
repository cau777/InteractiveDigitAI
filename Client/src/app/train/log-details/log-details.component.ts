import {AfterViewInit, Component, ElementRef, Input, ViewChild} from '@angular/core';
import {Log} from "../logs-view/logs-view.component";

@Component({
  selector: 'app-log-details',
  templateUrl: './log-details.component.html',
  styleUrls: ['./log-details.component.scss']
})
export class LogDetailsComponent implements AfterViewInit {
    @Input()
    public name!: string;
    
    @Input()
    public log!: Log;
    
    @Input()
    public expanded = false;
    
    @ViewChild("tbody")
    private tbody?: ElementRef<HTMLElement>;
    
    public ngAfterViewInit(): void {
        if (this.tbody === undefined) return;
        this.tbody.nativeElement.scroll(0, this.tbody.nativeElement.scrollHeight);
    }
}
