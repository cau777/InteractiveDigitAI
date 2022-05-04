import {Component} from '@angular/core';
import {animate, state, style, transition, trigger} from "@angular/animations";

@Component({
    selector: 'app-drawer',
    templateUrl: './drawer.component.html',
    styleUrls: ['./drawer.component.scss'],
    animations: [
        trigger('sidenavAnimationIsExpanded', [
            state('true', style({
                width: '200px'
            })),
            state('false', style({
                width: '64px'
            })),
            transition('false <=> true', animate('100ms ease'))
        ])
    ]
})
export class DrawerComponent {
    public toggled = false;
    
    public animating = false;
    
    public start() {
        this.animating = true;
        this.tick();
    }
    
    public done() {
        this.animating = false;
    }
    
    public tick() {
        if (this.animating) requestAnimationFrame(() => this.tick());
    }
}
