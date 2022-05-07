import {EventEmitter, Inject, Injectable} from '@angular/core';
import {DOCUMENT} from "@angular/common";

export type Position = { x: number, y: number };

function getMousePosition(e: MouseEvent): Position {
    return {x: e.clientX, y: e.clientY};
}

function getTouchPosition(e: TouchEvent): Position {
    return {x: e.targetTouches[0].clientX, y: e.targetTouches[0].clientY};
}

@Injectable({
    providedIn: 'root'
})
export class TouchService {
    public dragEmitter: EventEmitter<Position>;
    public touchEndEmitter: EventEmitter<void>;
    private position: Position;
    private dragging = false;
    
    public constructor(@Inject(DOCUMENT) private document: Document) {
        this.position = {x: 0, y: 0};
        this.dragEmitter = new EventEmitter<Position>();
        this.touchEndEmitter = new EventEmitter<void>();
        
        document.addEventListener("mousedown", e => {
            this.position = getMousePosition(e);
            this.touchStart();
        });
        document.addEventListener("touchstart", e => {
            this.position = getTouchPosition(e);
            this.touchStart();
        });
        
        document.addEventListener("mousemove", e => {
            this.position = getMousePosition(e);
            this.touchMove();
        });
        document.addEventListener("touchmove", e => {
            this.position = getTouchPosition(e);
            this.touchMove();
        });
        
        document.addEventListener("mouseup", () => {
            this.touchEnd();
        });
        document.addEventListener("touchend", () => {
            this.touchEnd();
        });
        
        document.addEventListener("mouseleave", () => {
            this.touchEnd();
        });
        document.addEventListener("touchcancel", () => {
            this.touchEnd();
        });
    }
    
    private touchStart() {
        this.dragging = true;
        this.dragEmitter.emit(this.position);
    }
    
    private touchMove() {
        if (this.dragging) this.dragEmitter.emit(this.position);
    }
    
    private touchEnd() {
        this.dragging = false;
        this.touchEndEmitter.emit();
    }
}
