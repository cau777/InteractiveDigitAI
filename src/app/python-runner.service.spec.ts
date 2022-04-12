import { TestBed } from '@angular/core/testing';

import { PythonRunnerService } from './python-runner.service';

describe('PythonRunnerService', () => {
  let service: PythonRunnerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PythonRunnerService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
