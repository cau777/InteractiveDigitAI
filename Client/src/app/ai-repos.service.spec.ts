import { TestBed } from '@angular/core/testing';

import { AiReposService } from './ai-repos.service';

describe('AiReposService', () => {
  let service: AiReposService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AiReposService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
