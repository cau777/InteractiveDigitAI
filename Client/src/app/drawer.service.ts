import {Injectable} from '@angular/core';

@Injectable({
    providedIn: 'root'
})
export class DrawerService {
    public static readonly MinWidth = 56;
    public static readonly MaxWidth = 200;
    
    public expanded = false;
    
    public get currentWidth() {
        return this.expanded ? DrawerService.MaxWidth : DrawerService.MinWidth;
    }
}
