import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {Position, TouchService} from "../../touch.service";
import {DrawerService} from "../../drawer.service";
import {PythonRunnerService} from "../../python-runner.service";
import {toNpArrayLiteral} from "../../pyodide/python-literals-conversions";

@Component({
    selector: 'app-digit-recognition-main',
    templateUrl: './digit-recognition-main.component.html',
    styleUrls: ['./digit-recognition-main.component.scss']
})
export class DigitRecognitionMainComponent implements AfterViewInit {
    @ViewChild("canvasElement")
    private canvasElement!: ElementRef<HTMLCanvasElement>;
    
    @ViewChild("resizeCanvasElement")
    private resizeCanvasElement!: ElementRef<HTMLCanvasElement>;
    
    public busy = false;
    private prevPos?: Position;
    
    
    public constructor(private touchService: TouchService, 
                       private drawerService: DrawerService,
                       private pythonRunner: PythonRunnerService) {}
    
    public ngAfterViewInit(): void {
        let canvas = this.canvasElement.nativeElement;
        let ctx = canvas.getContext("2d")!;
    
        this.touchService.dragEmitter.subscribe(({x, y}) => {
            let currX = x - canvas.offsetLeft - this.drawerService.currentWidth;
            let currY = y - canvas.offsetTop;
        
            if (this.prevPos !== undefined) {
                ctx.beginPath();
                ctx.moveTo(this.prevPos.x, this.prevPos.y);
                ctx.lineTo(currX, currY);
                ctx.lineWidth = 5;
                ctx.strokeStyle = "red"
                ctx.stroke();
                ctx.closePath();
            }
        
            this.prevPos = {x: currX, y: currY};
        });
        
        this.touchService.touchEndEmitter.subscribe(() => {
            this.prevPos = undefined;
        });
    }
    
    public clear() {
        this.canvasElement.nativeElement.getContext("2d")!.clearRect(0, 0, 96, 96);
    }
    
    public async predict() {
        console.log("Predicting");
        this.busy = true;
        await this.pythonRunner.loadScript("digit_recognition");
        let image = this.getResizedImage();
        
        await this.pythonRunner.run(`instance.eval(${toNpArrayLiteral(image, [28, 28])})`);
        
        this.busy = false;
    }
    
    private getResizedImage() {
        let canvas = this.canvasElement.nativeElement;
        
        let resizeCanvas = this.resizeCanvasElement.nativeElement;
        let resizeContext = resizeCanvas.getContext("2d")!;
        resizeContext.drawImage(canvas, 0, 0, 28, 28);
        
        let img = resizeContext.getImageData(0, 0, 28, 28, {colorSpace: "srgb"});
        return img.data.filter((value, index) => index % 4 === 0);
    }
}
