import {AfterViewInit, Component, ElementRef, Inject, OnDestroy, ViewChild} from '@angular/core';
import {Position, TouchService} from "../../touch.service";
import {DrawerService} from "../../drawer.service";
import {PythonRunnerService} from "../../python-runner.service";
import {AiReposService} from "../../ai-repos.service";
import {DOCUMENT} from "@angular/common";

/**
 * @link https://stackoverflow.com/questions/50393418/what-happens-if-i-dont-test-passive-event-listeners-support
 */
export function isPassiveEnabled() {
    let result = false;
    try {
        // Test via a getter in the options object to see if the passive property is accessed
        let opts = Object.defineProperty({}, 'passive', {
            get: function () {
                result = true;
            }
        });
        
        let name: any = "testPassive";
        let listener: any = null;
        
        window.addEventListener(name, listener, opts);
        window.removeEventListener(name, listener, opts);
    } catch (e) {}
    
    return result;
}

const passiveOptions: any = isPassiveEnabled() ? {passive: false} : false;

function formatPredictions(predictions: number[]) {
    predictions = predictions.map(Math.exp);
    let sum = predictions.reduce((prev, val) => prev + val);
    let labels = predictions.map((val, index) => [index, val / sum]);
    labels.sort((a, b) => b[1] - a[1]);
    
    let possible = labels.filter(([, val]) => val > 0.15).map(([index,]) => index);
    
    if (possible.length === 0) return "I have no idea";
    if (possible.length === 1) return "I'm certain this is a " + possible[0];
    
    return "I think this is a " + possible[0] + " but it could also be " + possible.slice(1).join(", ");
}

@Component({
    selector: 'app-digit-recognition-main',
    templateUrl: './digit-recognition-main.component.html',
    styleUrls: ['./digit-recognition-main.component.scss']
})
export class DigitRecognitionMainComponent implements AfterViewInit, OnDestroy {
    @ViewChild("canvasElement")
    private canvasElement!: ElementRef<HTMLCanvasElement>;
    
    @ViewChild("resizeCanvasElement")
    private resizeCanvasElement!: ElementRef<HTMLCanvasElement>;
    
    public state: "idle" | "loading" | "predicting" = "idle";
    public result?: string;
    private prevPos?: Position;
    private touchStartedInCanvas = false;
    
    public constructor(private touchService: TouchService,
                       private drawerService: DrawerService,
                       private pythonRunner: PythonRunnerService,
                       private aiReposService: AiReposService,
                       @Inject(DOCUMENT) private document: Document) {
        this.preventMobileScrolling = this.preventMobileScrolling.bind(this);
        window.addEventListener("touchmove", this.preventMobileScrolling, passiveOptions);
    }
    
    public ngOnDestroy() {
        window.removeEventListener("touchmove", this.preventMobileScrolling, passiveOptions);
    }
    
    public async ngAfterViewInit() {
        let canvas = this.canvasElement.nativeElement;
        
        canvas.addEventListener("touchstart", () => {
            this.touchStartedInCanvas = true;
        });
        canvas.addEventListener("mousedown", () => {
            this.touchStartedInCanvas = true;
        });
        
        let ctx = canvas.getContext("2d")!;
        
        this.touchService.dragEmitter.subscribe(({x, y}) => {
            if (!this.touchStartedInCanvas) return;
            
            let rect = canvas.getBoundingClientRect();
            let currX = x - rect.left;
            let currY = y - rect.top;
            
            if (this.prevPos !== undefined) {
                ctx.beginPath();
                ctx.moveTo(this.prevPos.x, this.prevPos.y);
                ctx.lineTo(currX, currY);
                ctx.lineWidth = 6;
                ctx.strokeStyle = "black";
                ctx.shadowColor = "black";
                ctx.shadowBlur = 4;
                ctx.lineCap = "round";
                ctx.lineJoin = "round";
                ctx.stroke();
                ctx.closePath();
            }
            
            this.prevPos = {x: currX, y: currY};
        });
        
        this.touchService.touchEndEmitter.subscribe(() => {
            this.prevPos = undefined;
            this.touchStartedInCanvas = false;
        });
    }
    
    public clear() {
        this.canvasElement.nativeElement.getContext("2d")!.clearRect(0, 0, 112, 112);
    }
    
    public async predict() {
        console.log("Loading");
        this.state = "loading";
        await this.pythonRunner.loadScript("digit_recognition");
        await this.aiReposService.loadModel("digit_recognition", this.pythonRunner);
        
        console.log("Predicting");
        this.state = "predicting";
        let image = this.getResizedImage();
        let predictions = await this.pythonRunner.run("instance.eval()", {inputs: image});
        this.result = formatPredictions(predictions);
        this.state = "idle";
    }
    
    private getResizedImage() {
        let canvas = this.canvasElement.nativeElement;
        
        let resizeCanvas = this.resizeCanvasElement.nativeElement;
        let resizeContext = resizeCanvas.getContext("2d")!;
        
        resizeContext.clearRect(0, 0, 28, 28);
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
        
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        let pixels = img.data.filter((value, index) => index % 4 === 3);
        
        return Array.from(pixels.values()).map(o => o / 255.0);
    }
    
    private preventMobileScrolling(e: TouchEvent) {
        if (this.touchStartedInCanvas) {
            e.preventDefault();
        }
    }
}
