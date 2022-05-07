import {Component, Input} from '@angular/core';

export type Log = { lines: LogLine[] };
export type LogLine = { content: string, isError: boolean };

@Component({
  selector: 'app-logs-view',
  templateUrl: './logs-view.component.html',
  styleUrls: ['./logs-view.component.scss']
})
export class LogsViewComponent {
    @Input()
    public entries!: [string, Log][];
}
