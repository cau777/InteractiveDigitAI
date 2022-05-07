import {Component} from '@angular/core';
import {animate, state, style, transition, trigger} from "@angular/animations";
import {DrawerService} from "../drawer.service";

@Component({
    selector: 'app-drawer',
    templateUrl: './drawer.component.html',
    styleUrls: ['./drawer.component.scss'],
    animations: [
        trigger('sidenavAnimationIsExpanded', [
            state('true', style({
                width: DrawerService.MaxWidth + "px"
            })),
            state('false', style({
                width: DrawerService.MinWidth + "px"
            })),
            transition('false <=> true', animate('100ms ease'))
        ])
    ]
})
export class DrawerComponent {
    public animating = false;
    
    public constructor(private drawerService: DrawerService) {
    }
    
    public toggle() {
        this.drawerService.expanded = !this.drawerService.expanded;
    }
    
    public get toggled() {
        return this.drawerService.expanded;
    }
    
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
