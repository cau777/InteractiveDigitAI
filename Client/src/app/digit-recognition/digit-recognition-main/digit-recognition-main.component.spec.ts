import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DigitRecognitionMainComponent } from './digit-recognition-main.component';

describe('DigitRecognitionMainComponent', () => {
  let component: DigitRecognitionMainComponent;
  let fixture: ComponentFixture<DigitRecognitionMainComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DigitRecognitionMainComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DigitRecognitionMainComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
