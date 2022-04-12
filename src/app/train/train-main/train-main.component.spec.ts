import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainMainComponent } from './train-main.component';

describe('TrainMainComponent', () => {
  let component: TrainMainComponent;
  let fixture: ComponentFixture<TrainMainComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TrainMainComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TrainMainComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
